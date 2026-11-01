import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createApp, recordBytes } from "./app.mjs";

const meta = { n: 3, nx: 2, ny: 2, nb: 1, layout: { kriging: 1, c_cat: 4, std_blocks: 4, E_blocks: 4 } };

async function withServer(state, fn, inferenceUrl = "http://127.0.0.1:9") {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "airtrace-"));
  fs.writeFileSync(path.join(dataDir, "real_meta.json"), JSON.stringify(meta));
  const size = recordBytes(meta);
  const bin = Buffer.alloc(size * meta.n);
  for (let i = 0; i < meta.n; i++) bin.writeUInt16LE(100 + i, i * size);
  fs.writeFileSync(path.join(dataDir, "real_fields.bin"), bin);
  const server = createApp({ dataDir, inferenceUrl, inferenceState: () => state }).listen(0, "127.0.0.1");
  await new Promise((r) => server.once("listening", r));
  try {
    await fn(`http://127.0.0.1:${server.address().port}`);
  } finally {
    server.close();
  }
}

test("record size follows the layout", () => assert.equal(recordBytes(meta), 2 * (5 * 4 + 8)));

test("window endpoint returns the right record and rejects bad indices", () =>
  withServer("ready", async (base) => {
    const buf = Buffer.from(await (await fetch(`${base}/api/real/window/2`)).arrayBuffer());
    assert.equal(buf.length, recordBytes(meta));
    assert.equal(buf.readUInt16LE(0), 102);
    assert.equal((await fetch(`${base}/api/real/window/3`)).status, 404);
    assert.equal((await fetch(`${base}/api/real/window/abc`)).status, 404);
  }));

test("lab calls report loading, then a dead service, as errors", async () => {
  await withServer("starting", async (base) => assert.equal((await fetch(`${base}/api/lab/scenarios`)).status, 503));
  await withServer("ready", async (base) => assert.equal((await fetch(`${base}/api/lab/scenarios`)).status, 502));
});

test("a missing data file gives a readable error", () =>
  withServer("ready", async (base) => {
    const r = await fetch(`${base}/api/map`);
    assert.equal(r.status, 404);
    assert.match((await r.json()).error, /phase7_precompute/);
  }));

test("lab requests reach the Python service with their body", async () => {
  const seen = [];
  const py = (await import("node:http")).createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      seen.push([req.method, req.url, body]);
      res.setHeader("content-type", "application/json");
      res.end('{"ok":true}');
    });
  }).listen(0, "127.0.0.1");
  await new Promise((r) => py.once("listening", r));
  try {
    await withServer("ready", async (base) => {
      const r = await fetch(`${base}/api/lab/predict`, { method: "POST", headers: { "content-type": "application/json" }, body: '{"scenario":1}' });
      assert.deepEqual(await r.json(), { ok: true });
    }, `http://127.0.0.1:${py.address().port}`);
    assert.deepEqual(seen[0], ["POST", "/lab/predict", '{"scenario":1}']);
  } finally {
    py.close();
  }
});
