import { spawn, type Subprocess } from "bun";
import { join } from "path";

const PORT = 3000;
const VOXTRAL_BIN = join(import.meta.dir, "..", "voxtral.c", "voxtral");
const MODEL_DIR = join(import.meta.dir, "..", "voxtral.c", "voxtral-model");
const PUBLIC_DIR = join(import.meta.dir, "public");

// Track active voxtral process per WebSocket
const sessions = new Map<object, { proc: Subprocess<"pipe", "pipe", "pipe">; alive: boolean }>();

const server = Bun.serve({
  port: PORT,
  hostname: "0.0.0.0",

  async fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket upgrade
    if (url.pathname === "/ws") {
      if (server.upgrade(req)) return;
      return new Response("WebSocket upgrade failed", { status: 400 });
    }

    // Serve static files
    let filePath = url.pathname === "/" ? "/index.html" : url.pathname;
    const file = Bun.file(join(PUBLIC_DIR, filePath));
    if (await file.exists()) {
      return new Response(file);
    }
    return new Response("Not found", { status: 404 });
  },

  websocket: {
    open(ws) {
      console.log("WebSocket connected, spawning voxtral...");

      const proc = spawn({
        cmd: [VOXTRAL_BIN, "-d", MODEL_DIR, "--stdin", "--silent", "-I", "2.0"],
        stdin: "pipe",
        stdout: "pipe",
        stderr: "pipe",
        env: { ...Bun.env, OPENBLAS_NUM_THREADS: "8", OMP_NUM_THREADS: "8" },
      });

      const session = { proc, alive: true };
      sessions.set(ws, session);

      // Stream stdout → WebSocket
      (async () => {
        const reader = proc.stdout.getReader();
        const decoder = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value, { stream: true });
            if (text && session.alive) {
              ws.sendText(text);
            }
          }
        } catch (e) {
          // Process ended or WebSocket closed
        }
        // Signal end of transcription
        if (session.alive) {
          ws.sendText("\n[END]");
        }
      })();

      // Log stderr for debugging
      (async () => {
        const reader = proc.stderr.getReader();
        const decoder = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = decoder.decode(value, { stream: true });
            if (text) process.stderr.write(`[voxtral] ${text}`);
          }
        } catch (e) {
          // Ignore
        }
      })();

      console.log("voxtral process started (PID:", proc.pid, ")");
    },

    message(ws, message) {
      const session = sessions.get(ws);
      if (!session) return;

      // Binary message = raw s16le PCM audio data
      if (message instanceof ArrayBuffer || message instanceof Uint8Array) {
        const data = message instanceof Uint8Array ? message : new Uint8Array(message);
        try {
          session.proc.stdin.write(data);
        } catch (e) {
          // stdin closed
        }
      }
    },

    close(ws) {
      console.log("WebSocket closed");
      const session = sessions.get(ws);
      if (session) {
        session.alive = false;
        try {
          session.proc.stdin.end();
        } catch (e) {
          // Already closed
        }
        // Give voxtral a moment to flush, then kill
        setTimeout(() => {
          try {
            session.proc.kill();
          } catch (e) {
            // Already dead
          }
        }, 5000);
        sessions.delete(ws);
      }
    },
  },
});

console.log(`Voxtral Web listening on http://0.0.0.0:${PORT}`);
console.log(`Binary: ${VOXTRAL_BIN}`);
console.log(`Model:  ${MODEL_DIR}`);
