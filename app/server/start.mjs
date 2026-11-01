// `npm start`: launch the Python inference service, then serve the app on http://localhost:3000.
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createApp } from "./app.mjs";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(appDir, "..");
const PORT = Number(process.env.PORT ?? 3000);
const INFER_PORT = Number(process.env.AIRTRACE_INFER_PORT ?? 8765);
const inferenceUrl = `http://127.0.0.1:${INFER_PORT}`;
const webDir = path.join(appDir, "web", "dist");

if (!fs.existsSync(webDir)) {
  console.log("Building the web UI (first run)...");
  const b = spawnSync("npm", ["run", "build"], { cwd: appDir, stdio: "inherit", shell: true });
  if (b.status !== 0) process.exit(1);
}

const python = process.env.AIRTRACE_PYTHON
  ?? [path.join(root, ".venv", "Scripts", "python.exe"), path.join(root, ".venv", "bin", "python")].find(fs.existsSync)
  ?? "python";

let state = "starting";
// CUDA hidden on purpose: the demo must run on any laptop CPU
const child = spawn(python, ["-m", "uvicorn", "app.inference.service:app", "--host", "127.0.0.1", "--port", String(INFER_PORT), "--log-level", "warning"],
  { cwd: root, env: { ...process.env, CUDA_VISIBLE_DEVICES: "" }, stdio: ["ignore", "inherit", "inherit"] });
child.on("exit", (code) => {
  state = "down";
  console.error(`Inference service exited (code ${code}). The Scenario lab is unavailable; other screens still work.`);
});

const poll = setInterval(async () => {
  if (state === "down") return clearInterval(poll);
  try {
    if ((await fetch(`${inferenceUrl}/health`)).ok) {
      state = "ready";
      clearInterval(poll);
      console.log("Model service ready.");
    }
  } catch { /* still loading */ }
}, 1000);

const server = createApp({ dataDir: path.join(appDir, "data"), webDir, inferenceUrl, inferenceState: () => state })
  .listen(PORT, "127.0.0.1", () => console.log(`AirTrace is running at http://localhost:${PORT}`));

function stop() {
  child.kill();
  server.close(() => process.exit(0));
}
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
