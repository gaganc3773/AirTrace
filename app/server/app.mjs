// Express app: serves the built web UI, the precomputed data pack, and proxies lab calls to the Python service.
import express from "express";
import fs from "node:fs";
import path from "node:path";

const JSON_FILES = { map: "map.json", evidence: "evidence.json", hidden: "hidden_story.json", "real/meta": "real_meta.json" };

/** Bytes of one record in real_fields.bin, from the layout in real_meta.json. */
export function recordBytes(meta) {
  const full = meta.nx * meta.ny;
  const blocks = meta.nb * meta.nb;
  return 2 * ((meta.layout.kriging + meta.layout.c_cat) * full + (meta.layout.std_blocks + meta.layout.E_blocks) * blocks);
}

/**
 * @param {object} opts
 * @param {string} opts.dataDir   folder holding the data pack
 * @param {string} [opts.webDir]  built web UI, when present
 * @param {string} opts.inferenceUrl  base URL of the Python service
 * @param {() => string} opts.inferenceState  "starting" | "ready" | "down"
 */
export function createApp({ dataDir, webDir, inferenceUrl, inferenceState }) {
  const app = express();
  app.use(express.json({ limit: "200kb" }));

  for (const [route, file] of Object.entries(JSON_FILES)) {
    app.get(`/api/${route}`, (_req, res) => {
      const p = path.join(dataDir, file);
      if (!fs.existsSync(p)) return res.status(404).json({ error: `${file} is missing. Run scripts/phase7_precompute.py` });
      res.type("application/json").sendFile(p);
    });
  }

  let meta = null;
  app.get("/api/real/window/:i", (req, res) => {
    const binPath = path.join(dataDir, "real_fields.bin");
    meta ??= JSON.parse(fs.readFileSync(path.join(dataDir, "real_meta.json"), "utf8"));
    const i = Number(req.params.i);
    if (!Number.isInteger(i) || i < 0 || i >= meta.n) return res.status(404).json({ error: "No such window" });
    const size = recordBytes(meta);
    const buf = Buffer.alloc(size);
    const fd = fs.openSync(binPath, "r");
    try {
      fs.readSync(fd, buf, 0, size, i * size);
    } finally {
      fs.closeSync(fd);
    }
    res.set("Cache-Control", "public, max-age=3600").type("application/octet-stream").send(buf);
  });

  app.get("/api/status", (_req, res) => res.json({ inference: inferenceState() }));

  app.use("/api/lab", async (req, res) => {
    if (inferenceState() !== "ready") return res.status(503).json({ error: "The model is still loading. Try again in a few seconds." });
    try {
      const r = await fetch(`${inferenceUrl}/lab${req.url}`, {
        method: req.method,
        headers: { "content-type": "application/json" },
        body: req.method === "GET" ? undefined : JSON.stringify(req.body),
      });
      res.status(r.status).type("application/json").send(Buffer.from(await r.arrayBuffer()));
    } catch {
      res.status(502).json({ error: "The model service isn't responding." });
    }
  });

  app.use("/api", (_req, res) => res.status(404).json({ error: "Not found" }));
  if (webDir && fs.existsSync(webDir)) {
    app.use(express.static(webDir));
    app.use((_req, res) => res.sendFile(path.join(webDir, "index.html")));
  }
  return app;
}
