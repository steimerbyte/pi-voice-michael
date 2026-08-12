import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { KokoroTTS } from "kokoro-js";
import { env as hfEnv } from "@huggingface/transformers";
import { spawn } from "node:child_process";
import { mkdtemp, rm, stat, readdir, writeFile, mkdir } from "node:fs/promises";
import { existsSync, createWriteStream } from "node:fs";
import { join } from "node:path";
import { tmpdir, homedir } from "node:os";
import { platform } from "node:process";
import { createRequire } from "node:module";
import https from "node:https";

const DEFAULT_VOICE = "am_michael";
// kokoro-js 1.2.1 requires this specific repo id
const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

// ─── Self-managed cache location ─────────────────────────────────────────
// Stable, predictable location next to pi's agent cache. Plugin owns this
// directory entirely — no scattering in node_modules/.cache.
//
// Layout (required by @huggingface/transformers FileCache):
//   ~/.pi/agent/cache/pi-voice-michael/onnx-community/Kokoro-82M-v1.0-ONNX/
//     ├── config.json
//     ├── tokenizer.json
//     ├── tokenizer_config.json
//     └── onnx/
//         └── model_quantized.onnx    (~89 MB)
const PLUGIN_CACHE_DIR = join(homedir(), ".pi", "voice");
const MODEL_DIR = join(PLUGIN_CACHE_DIR, "onnx-community", "Kokoro-82M-v1.0-ONNX");
const ONNX_DIR = join(MODEL_DIR, "onnx");

// Legacy cache paths from earlier versions — auto-migrate on first run
const LEGACY_CACHE_PATHS = [
  join(homedir(), ".pi", "agent", "cache", "pi-voice-michael"),
  join(homedir(), ".cache", "huggingface", "hub", "models--onnx-community--Kokoro-82M-v1.0-ONNX"),
  join(homedir(), ".cache", "huggingface", "models--onnx-community--Kokoro-82M-v1.0-ONNX"),
];

// On startup, if the new cache is missing but a legacy one exists, symlink it
function migrateLegacyCache(): void {
  if (existsSync(ONNX_DIR) && existsSync(MODEL_DIR)) return; // already migrated
  for (const legacy of LEGACY_CACHE_PATHS) {
    if (!existsSync(legacy)) continue;
    try {
      // Map legacy onnx-community path → our onnx-community path
      const legacyRepoDir = legacy.includes("models--")
        ? join(legacy, "snapshots")
        : join(legacy, "onnx-community", "Kokoro-82M-v1.0-ONNX");
      if (!existsSync(legacyRepoDir)) continue;
      mkdir(join(PLUGIN_CACHE_DIR, "onnx-community"), { recursive: true });
      // Try hard-link or symlink; fall back to copy
      const target = MODEL_DIR;
      try {
        mkdir(target, { recursive: true });
        for (const f of ["model_quantized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json"]) {
          const src = join(legacyRepoDir, f);
          const dst = f === "model_quantized.onnx" ? join(ONNX_DIR, f) : join(MODEL_DIR, f);
          if (!existsSync(src) || existsSync(dst)) continue;
          // Hard-link if possible (instant, no extra disk space)
          try {
            require("node:fs").linkSync(src, dst);
          } catch {
            // Fall back to copy
            const { copyFile } = require("node:fs/promises");
            copyFile(src, dst).catch(() => {});
          }
        }
      } catch { /* swallow */ }
      return;
    } catch { /* try next */ }
  }
}
// Run migration at module load (sync, fast)
migrateLegacyCache();

const MODEL_FILES: Record<string, { path: string; expectedBytes: number }> = {
  "onnx/model_quantized.onnx": { path: join(ONNX_DIR, "model_quantized.onnx"), expectedBytes: 90_000_000 },
  "config.json": { path: join(MODEL_DIR, "config.json"), expectedBytes: 44 },
  "tokenizer.json": { path: join(MODEL_DIR, "tokenizer.json"), expectedBytes: 4608 },
  "tokenizer_config.json": { path: join(MODEL_DIR, "tokenizer_config.json"), expectedBytes: 113 },
};

// Force @huggingface/transformers to use our cache directory BEFORE Kokoro
// instantiates its pipeline. This prevents the library from scattering files
// in node_modules/.cache directories. Must run at module-load time.
hfEnv.cacheDir = PLUGIN_CACHE_DIR;
// By default refuse remote downloads — plugin is 100% offline-first
hfEnv.allowRemoteModels = process.env.PI_VOICE_ONLINE === "1";
hfEnv.allowLocalModels = true;

// ─── Helpers ─────────────────────────────────────────────────────────────
function pickPlayer(): { cmd: string; args: (path: string) => string[] } | null {
  const p = platform;
  if (p === "darwin") return { cmd: "afplay", args: (f) => [f] };
  if (p === "win32") {
    return {
      cmd: "powershell",
      args: (f) => [
        "-NoProfile", "-Command",
        `Add-Type -AssemblyName PresentationCore; (New-Object System.Media.SoundPlayer '${f.replace(/'/g, "''")}').PlaySync()`,
      ],
    };
  }
  return {
    cmd: "sh",
    args: (f) => [
      "-c",
      `command -v paplay >/dev/null && paplay '${f}' || (command -v pw-play >/dev/null && pw-play '${f}' || aplay -q '${f}')`,
    ],
  };
}

function playWav(wavPath: string): Promise<void> {
  const player = pickPlayer();
  if (!player) return Promise.reject(new Error(`No audio player for platform ${platform}`));
  return new Promise((resolve, reject) => {
    const proc = spawn(player.cmd, player.args(wavPath), { stdio: "ignore" });
    proc.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`audio player exited ${code}`))));
    proc.on("error", reject);
  });
}

async function detectPlayers(): Promise<{ name: string; available: boolean }[]> {
  const checks: { name: string; cmd: string }[] =
    platform === "darwin"
      ? [{ name: "afplay", cmd: "afplay" }]
      : platform === "win32"
      ? [{ name: "powershell", cmd: "powershell" }]
      : [
          { name: "paplay (PulseAudio)", cmd: "paplay" },
          { name: "pw-play (PipeWire)", cmd: "pw-play" },
          { name: "aplay (ALSA)", cmd: "aplay" },
          { name: "ffplay (ffmpeg fallback)", cmd: "ffplay" },
        ];
  return Promise.all(
    checks.map(async ({ name, cmd }) => {
      try {
        const res = await new Promise<{ status: number | null }>((resolve) => {
          const p = spawn("sh", ["-c", `command -v ${cmd}`], { stdio: "ignore" });
          p.on("exit", (code) => resolve({ status: code }));
          p.on("error", () => resolve({ status: -1 }));
        });
        return { name, available: res.status === 0 };
      } catch {
        return { name, available: false };
      }
    }),
  );
}

// ─── Self-managed model cache ────────────────────────────────────────────
async function ensureCacheDir(): Promise<void> {
  await mkdir(ONNX_DIR, { recursive: true });
  await mkdir(MODEL_DIR, { recursive: true });
}

async function isCacheComplete(): Promise<{
  complete: boolean;
  files: Record<string, { exists: boolean; size: number; expectedBytes: number; path: string }>;
}> {
  const files: Record<string, { exists: boolean; size: number; expectedBytes: number; path: string }> = {};
  for (const [name, info] of Object.entries(MODEL_FILES)) {
    if (existsSync(info.path)) {
      try {
        const s = await stat(info.path);
        files[name] = { exists: true, size: s.size, expectedBytes: info.expectedBytes, path: info.path };
      } catch {
        files[name] = { exists: false, size: 0, expectedBytes: info.expectedBytes, path: info.path };
      }
    } else {
      files[name] = { exists: false, size: 0, expectedBytes: info.expectedBytes, path: info.path };
    }
  }
  // A file is "complete" if it exists AND has at least 50% of expected size
  // (the ONNX file should be exactly ~89MB; JSONs are tiny)
  const complete = Object.values(files).every((f) => f.exists && f.size >= f.expectedBytes * 0.5);
  return { complete, files };
}

// Returns a precise error message if model is missing — tells user exactly where
// to put files for manual installation (the whole point of self-managed cache)
function missingModelError(): string {
  return (
    `Model not found in plugin cache. Required files:\n` +
    `  ${ONNX_DIR}/model_quantized.onnx    (~89 MB)\n` +
    `  ${MODEL_DIR}/config.json\n` +
    `  ${MODEL_DIR}/tokenizer.json\n` +
    `  ${MODEL_DIR}/tokenizer_config.json\n\n` +
    `To install manually:\n` +
    `  1. mkdir -p "${PLUGIN_CACHE_DIR}"\n` +
    `  2. From huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX download:\n` +
    `     - onnx/model_quantized.onnx  →  ${ONNX_DIR}/\n` +
    `     - config.json                →  ${MODEL_DIR}/\n` +
    `     - tokenizer.json             →  ${MODEL_DIR}/\n` +
    `     - tokenizer_config.json      →  ${MODEL_DIR}/\n` +
    `Or set PI_VOICE_ONLINE=1 to allow auto-download on first use.`
  );
}

// Download a single file via HTTPS with redirect following
function downloadFile(
  url: string,
  dest: string,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const follow = (u: string): void => {
      https.get(u, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          follow(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${u}`));
          return;
        }
        const total = parseInt(res.headers["content-length"] ?? "0", 10);
        let downloaded = 0;
        const out = createWriteStream(dest);
        res.on("data", (chunk: Buffer) => {
          downloaded += chunk.length;
          onProgress?.(downloaded, total);
        });
        res.pipe(out);
        out.on("finish", () => out.close(() => resolve()));
        out.on("error", reject);
        res.on("error", reject);
      }).on("error", reject);
    };
    follow(url);
  });
}

// Download model files from HuggingFace into our self-managed cache.
async function downloadModel(onProgress?: (msg: string, percent?: number) => void): Promise<void> {
  await ensureCacheDir();
  const { complete, files } = await isCacheComplete();
  if (complete) {
    onProgress?.("Model cache already complete", 100);
    return;
  }

  // Map relative file path to (URL, dest). ONNX file lives in /onnx/ subfolder;
  // JSON configs live at repo root.
  const downloads: { key: string; url: string; dest: string }[] = Object.entries(files)
    .filter(([, info]) => !info.exists)
    .map(([key, info]) => ({
      key,
      url: key.startsWith("onnx/")
        ? `https://huggingface.co/${MODEL_ID}/resolve/main/${key}`
        : `https://huggingface.co/${MODEL_ID}/resolve/main/${key.split("/").pop()}`,
      dest: info.path,
    }));

  let done = 0;
  for (const d of downloads) {
    done++;
    const basePct = Math.round((done / downloads.length) * 100);
    const label = d.key.split("/").pop() ?? d.key;
    onProgress?.(`Downloading ${label} (${done}/${downloads.length})…`, basePct - 5);
    await downloadFile(d.url, d.dest);
  }
  onProgress?.("Model download complete", 100);
}

// ─── TTS lifecycle ──────────────────────────────────────────────────────
async function tryModelLoad(): Promise<{
  ok: boolean;
  ms?: number;
  voices?: string[];
  error?: string;
  onnxVersion?: string;
}> {
  const t0 = Date.now();
  try {
    const m = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
    const voices = (m as unknown as { list_voices?: () => string[] }).list_voices?.() ?? [];
    let onnxVersion = "unknown";
    try {
      const req = createRequire(import.meta.url ?? __filename);
      const ortPkg = req("onnxruntime-node/package.json");
      onnxVersion = ortPkg.version;
    } catch { /* ignore */ }
    return { ok: true, ms: Date.now() - t0, voices, onnxVersion };
  } catch (err: unknown) {
    return { ok: false, error: (err as Error)?.message ?? String(err) };
  }
}

async function tryEndToEnd(voice: string): Promise<{
  ok: boolean;
  ms?: number;
  bytes?: number;
  error?: string;
}> {
  try {
    const tts = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
    const t0 = Date.now();
    const audio = await tts.generate("Voice test.", { voice } as unknown as Record<string, string>);
    const dir = await mkdtemp(join(tmpdir(), "pi-voice-doctor-"));
    const wavPath = join(dir, "test.wav");
    await audio.save(wavPath);
    const s = await stat(wavPath);
    await playWav(wavPath);
    rm(dir, { recursive: true, force: true }).catch(() => {});
    return { ok: true, ms: Date.now() - t0, bytes: s.size };
  } catch (err: unknown) {
    return { ok: false, error: (err as Error)?.message ?? String(err) };
  }
}

async function tryPkgVersion(name: string): Promise<string | undefined> {
  // Modern packages use `exports` field that blocks `require.resolve(name)`.
  // Strategy: try resolving from every relevant consumer (kokoro-js,
  // @huggingface/transformers), then pick the most-deeply-nested match —
  // that's the version Node would actually load when Kokoro delegates to
  // transformers (which has its own nested onnxruntime-node). Fallback:
  // walk node_modules from the plugin's directory.
  const anchors = ["kokoro-js", "@huggingface/transformers"];
  const candidates: { pkgDir: string; nestedness: number }[] = [];
  for (const anchor of anchors) {
    try {
      const req = createRequire(import.meta.url ?? __filename);
      const anchorPath = req.resolve(anchor);
      const anchorReq = createRequire(join(anchorPath, "..", "package.json"));
      const resolved = anchorReq.resolve(name);
      const parts = resolved.split("/");
      const nmIdx = parts.lastIndexOf("node_modules");
      if (nmIdx < 0) continue;
      const isScoped = parts[nmIdx + 1].startsWith("@");
      const offset = isScoped ? 3 : 2;
      const pkgDir = parts.slice(0, nmIdx + offset).join("/");
      const nestedness = (resolved.match(/node_modules/g) ?? []).length;
      candidates.push({ pkgDir, nestedness });
    } catch { /* exports field blocking — try walk fallback */ }
  }
  // Fallback: walk node_modules upward from this file's directory
  try {
    const here = (import.meta.url ?? __filename).replace(/^file:\/\//, "");
    let cursor = here.split("/").slice(0, -1).join("/");
    const isScoped = name.startsWith("@");
    const segs = isScoped ? name.split("/") : [name];
    while (cursor && cursor !== "/" && cursor !== ".") {
      const candidate = join(cursor, "node_modules", ...segs);
      if (existsSync(candidate)) {
        const nestedness = (candidate.match(/node_modules/g) ?? []).length;
        candidates.push({ pkgDir: candidate, nestedness });
      }
      cursor = cursor.split("/").slice(0, -1).join("/");
    }
  } catch { /* ignore */ }

  candidates.sort((a, b) => b.nestedness - a.nestedness);
  for (const c of candidates) {
    try {
      const { readFile } = await import("node:fs/promises");
      const pkg = JSON.parse(await readFile(join(c.pkgDir, "package.json"), "utf8"));
      return pkg.version as string;
    } catch { continue; }
  }
  return undefined;
}

// ─── Plugin ──────────────────────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let tts: KokoroTTS | null = null;
  let ttsLoading: Promise<KokoroTTS> | null = null;

  async function ensureModelReady(onUpdate?: (msg: string, percent?: number) => void): Promise<void> {
    onUpdate?.("Checking self-managed cache…", 0);
    await ensureCacheDir();
    const cache = await isCacheComplete();
    if (!cache.complete) {
      if (process.env.PI_VOICE_ONLINE === "1") {
        onUpdate?.("Cache incomplete — downloading model (~90 MB)…", 5);
        await downloadModel((msg, pct) => onUpdate?.(msg, pct));
        onUpdate?.("Model downloaded to plugin cache", 95);
      } else {
        // Offline-first: surface precise error so user knows exactly where to put files
        throw new Error(missingModelError());
      }
    } else {
      onUpdate?.("Model cache ready (offline)", 95);
    }
  }

  async function ensureTTS(onUpdate?: (msg: string, percent?: number) => void): Promise<KokoroTTS> {
    if (tts) return tts;
    if (!ttsLoading) {
      ttsLoading = (async () => {
        await ensureModelReady(onUpdate);
        const m = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
        tts = m;
        return m;
      })();
    }
    return ttsLoading;
  }

  async function speak(
    text: string,
    voice: string,
    onUpdate?: (msg: string, percent?: number) => void,
  ): Promise<{ ok: boolean; voice: string; text: string; file?: string; error?: string }> {
    const dir = await mkdtemp(join(tmpdir(), "pi-voice-"));
    const wavPath = join(dir, `speech-${Date.now()}.wav`);
    try {
      onUpdate?.("Initializing…", 5);
      const t0 = Date.now();
      const model = await ensureTTS(onUpdate);
      onUpdate?.(`Model ready in ${((Date.now() - t0) / 1000).toFixed(1)}s, generating speech…`, 70);

      const t1 = Date.now();
      const audio = await model.generate(text, { voice } as unknown as Record<string, string>);
      onUpdate?.(`Synthesized in ${((Date.now() - t1) / 1000).toFixed(1)}s, preparing playback…`, 90);

      await audio.save(wavPath);
      onUpdate?.("Playing through speakers…", 95);

      try {
        await playWav(wavPath);
      } finally {
        // Free the audio buffer + WAV file BEFORE returning
        try { (audio as unknown as { data?: Float32Array }).data = undefined; } catch {}
      }
      return { ok: true, voice, text, file: wavPath };
    } catch (err: unknown) {
      return { ok: false, voice, text, error: (err as Error)?.message ?? String(err) };
    } finally {
      // Always clean up temp WAV file (memory + disk)
      rm(dir, { recursive: true, force: true }).catch(() => {});
      // Hint GC: release model reference if idle for >5 minutes (cheap idle check)
      scheduleIdleUnload();
    }
  }

  // ─── Idle unload: release Kokoro after 5min of inactivity ────────────
  // Kokoro caches every voice it has ever used in an internal Map that is
  // never garbage-collected. After many voice switches, that map can hold
  // dozens of 510KB Float32Arrays (~2MB heap each). We force release the
  // whole Kokoro instance after 5min idle, then reload on next call.
  let idleTimer: NodeJS.Timeout | null = null;
  function scheduleIdleUnload(): void {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      const mem = process.memoryUsage();
      const rssMB = Math.round(mem.rss / 1024 / 1024);
      // Unload if either: (a) RSS > 500 MB or (b) we've been idle for the full
      // 5 minutes regardless. The latter is cheap insurance.
      tts = null;
      ttsLoading = null;
      if (global.gc) {
        try { global.gc(); } catch {}
      }
      idleTimer = null;
    }, 5 * 60 * 1000);
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify(
      `🔊 pi-voice-michael loaded. Cache: ${PLUGIN_CACHE_DIR.replace(homedir(), "~")}. Run /voice-doctor to verify.`,
      "info",
    );
    // Background pre-warm: ensure model is cached and loaded
    (async () => {
      try { await ensureTTS(); } catch { /* silent */ }
    })();
  });

  pi.on("session_shutdown", async () => {
    tts = null;
    ttsLoading = null;
  });

  // ─── Tool: voice_say_aloud ─────────────────────────────────────────────
  pi.registerTool({
    name: "voice_say_aloud",
    label: "Speak Aloud (am_michael)",
    description:
      "Convert text to speech and play it aloud through the user's speakers using the offline am_michael voice (Kokoro ONNX, US English male). Self-managed cache at ~/.pi/agent/cache/pi-voice-michael/. Use this when the user explicitly asks you to speak, or when a verbal response is appropriate. Pass plain conversational text — no markdown, no code blocks, no URLs. First call may take ~30s for model download/init; subsequent calls are fast (~3s). If setup is broken, run /voice-doctor first.",
    parameters: Type.Object({
      text: Type.String({
        description:
          "The text to speak aloud. Plain conversational English, no formatting. Keep it short (1–2 sentences ideal).",
      }),
      voice: Type.Optional(
        Type.String({
          description:
            "Optional voice override. Defaults to am_michael. Other voices: am_fenrir, am_puck, bm_george, af_heart, af_bella, etc.",
        }),
      ),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const { text, voice } = params as { text: string; voice?: string };
      const v = voice || DEFAULT_VOICE;
      const report = (msg: string, percent?: number) => {
        if (!onUpdate) return;
        try {
          onUpdate({
            content: [{ type: "text", text: `🔊 ${msg}` }],
            details: { phase: msg, percent: percent ?? null },
          });
        } catch { /* ignore */ }
      };
      report("Starting TTS pipeline…", 0);
      const r = await speak(text, v, report);
      if (r.ok) {
        return {
          content: [{ type: "text", text: `🔊 Spoke aloud (${r.voice}): "${text}"` }],
          details: { ...r, percent: 100 },
        };
      }
      return {
        content: [
          {
            type: "text",
            text: `❌ TTS playback failed: ${r.error}\nRun /voice-doctor to diagnose.`,
          },
        ],
        details: r,
        isError: true,
      };
    },
  });

  // ─── Command: /say ────────────────────────────────────────────────────
  pi.registerCommand("say", {
    description: "Speak text aloud using the offline am_michael voice. Usage: /say <text> [voice]",
    async handler(args, ctx) {
      const trimmed = args.trim();
      if (!trimmed) {
        ctx.ui.notify("Usage: /say <text> [voice]", "warning");
        return;
      }
      const quotedMatch = trimmed.match(/^"([^"]+)"(?:\s+(\S+))?$/);
      let text: string;
      let voice: string | undefined;
      if (quotedMatch) {
        text = quotedMatch[1];
        voice = quotedMatch[2];
      } else {
        const tokens = trimmed.split(/\s+/);
        if (tokens.length >= 2 && /^(am|af|bm|bf|jm)_/.test(tokens[tokens.length - 1])) {
          voice = tokens.pop();
        }
        text = tokens.join(" ");
      }
      ctx.ui.setStatus("pi-voice", "🔊 Initializing…");
      const r = await speak(text, voice || DEFAULT_VOICE, (msg, pct) => {
        ctx.ui.setStatus("pi-voice", `🔊 ${msg}${pct != null ? ` ${pct}%` : ""}`);
      });
      ctx.ui.setStatus("pi-voice", "");
      if (r.ok) ctx.ui.notify(`🔊 Spoke (${r.voice}): "${text}"`, "info");
      else ctx.ui.notify(`❌ TTS failed: ${r.error}`, "error");
    },
  });

  // ─── Command: /voice-doctor ─────────────────────────────────────────────
  // Design principles:
  //   1. ONE notify per check — short, scannable, never truncated.
  //   2. Full detail lines → widget as a compact one-line-per-check list.
  //   3. No markdown sections, no multi-line strings in notify.
  pi.registerCommand("voice-doctor", {
    description:
      "Diagnose pi-voice-michael setup — verifies full offline capability using self-managed cache",
    async handler(_args, ctx) {
      ctx.ui.setStatus("voice-doctor", "Running…");
      const checks: { ok: boolean; name: string; detail: string }[] = [];

      // ── 1. Audio player ──────────────────────────────────────────────
      const players = await detectPlayers();
      const availPlayer = players.find((p) => p.available);
      const playerOk = !!availPlayer;
      if (playerOk) {
        ctx.ui.notify(`1/5 ✓ Audio: ${availPlayer.name} available`, "info");
      } else {
        const missing = players.filter((p) => !p.available).map((p) => p.name).join(", ");
        ctx.ui.notify(`1/5 ❌ Audio: none found (tried: ${missing})`, "error");
      }
      checks.push({
        ok: playerOk,
        name: "Audio player",
        detail: playerOk ? availPlayer.name : `none (tried: ${players.map((p) => p.name).join(", ")})`,
      });

      // ── 2. NPM dependencies ────────────────────────────────────────────
      const [kokoroVer, transVer, ortVer, piCodingVer] = await Promise.all([
        tryPkgVersion("kokoro-js"),
        tryPkgVersion("@huggingface/transformers"),
        tryPkgVersion("onnxruntime-node"),
        tryPkgVersion("@earendil-works/pi-coding-agent"),
      ]);
      const depsOk = !!(kokoroVer && transVer && ortVer && piCodingVer);
      let ortNote = "";
      if (ortVer) {
        const minor = parseInt(ortVer.split(".")[1], 10);
        if (minor < 20 || minor > 21) {
          ortNote = ` ⚠️ ort ${ortVer} may be incompatible (use ~1.21.0)`;
        }
      }
      ctx.ui.notify(
        `2/5 ${depsOk ? "✓" : "❌"} Deps: kokoro=${kokoroVer ?? "?"}, hf=${transVer ?? "?"}, ort=${ortVer ?? "?"}${ortNote}`,
        depsOk ? "info" : "error",
      );
      checks.push({
        ok: depsOk,
        name: "Dependencies",
        detail: `kokoro=${kokoroVer ?? "MISSING"}, hf=${transVer ?? "MISSING"}, ort=${ortVer ?? "MISSING"}${ortNote}`,
      });

      // ── 3. Self-managed cache ──────────────────────────────────────────
      await ensureCacheDir();
      const cache = await isCacheComplete();
      const cacheOk = cache.complete;
      const totalMB = Object.values(cache.files).reduce((s, f) => s + f.size, 0) / 1024 / 1024;
      const cachePath = PLUGIN_CACHE_DIR.replace(homedir(), "~");
      ctx.ui.notify(
        `3/5 ${cacheOk ? "✓" : "⚠️"} Cache: ${totalMB.toFixed(1)} MB at ${cachePath}${cacheOk ? "" : " — INCOMPLETE"}`,
        cacheOk ? "info" : "warning",
      );
      checks.push({
        ok: cacheOk,
        name: "Model cache",
        detail: `${totalMB.toFixed(1)} MB, ${cacheOk ? "complete" : "incomplete"}`,
      });

      // ── 4. Model load test ─────────────────────────────────────────────
      const loadResult = await tryModelLoad();
      const loadOk = loadResult.ok && (loadResult.voices?.includes(DEFAULT_VOICE) !== false);
      if (loadResult.ok) {
        ctx.ui.notify(
          `4/5 ✓ Model: loaded in ${(loadResult.ms! / 1000).toFixed(1)}s (ort ${loadResult.onnxVersion})${loadResult.voices ? `, ${loadResult.voices.length} voices` : ""}`,
          "info",
        );
      } else {
        ctx.ui.notify(`4/5 ❌ Model: ${loadResult.error?.split("\n")[0] ?? "load failed"}`, "error");
      }
      checks.push({
        ok: loadOk,
        name: "Model load",
        detail: loadResult.ok
          ? `${(loadResult.ms! / 1000).toFixed(1)}s, ort ${loadResult.onnxVersion}, ${loadResult.voices?.length ?? 0} voices`
          : `FAILED: ${loadResult.error?.split("\n")[0]}`,
      });

      // ── 5. End-to-end ─────────────────────────────────────────────────
      let e2eOk = false;
      if (!playerOk || !loadOk) {
        ctx.ui.notify("5/5 ⏭️ E2E: skipped (player or model check failed)", "warning");
        checks.push({ ok: false, name: "End-to-end", detail: "skipped (prerequisites failed)" });
      } else {
        const e2e = await tryEndToEnd(DEFAULT_VOICE);
        e2eOk = e2e.ok;
        if (e2e.ok) {
          ctx.ui.notify(
            `5/5 ✓ E2E: synthesized + played in ${(e2e.ms! / 1000).toFixed(1)}s (${(e2e.bytes! / 1024).toFixed(0)} KB)`,
            "info",
          );
        } else {
          ctx.ui.notify(`5/5 ❌ E2E: ${e2e.error?.split("\n")[0] ?? "failed"}`, "error");
        }
        checks.push({
          ok: e2eOk,
          name: "End-to-end",
          detail: e2e.ok
            ? `${(e2e.ms! / 1000).toFixed(1)}s, ${(e2e.bytes! / 1024).toFixed(0)} KB WAV`
            : `FAILED: ${e2e.error?.split("\n")[0]}`,
        });
      }

      // ── Summary widget ─────────────────────────────────────────────────
      const allOk = checks.every((c) => c.ok);
      const passed = checks.filter((c) => c.ok).length;
      const failed = checks.filter((c) => !c.ok).map((c) => c.name);

      const widgetLines = [
        `# /voice-doctor  ${passed}/5 passed`,
        "",
        ...checks.map((c) => `  ${c.ok ? "✓" : "✗"} [${c.name}] ${c.detail}`),
        "",
        allOk
          ? `🎉 Fully offline-capable. Cache: \`${cachePath}\``
          : `⚠️ Failed: ${failed.join(", ")}. See notify details above.`,
      ];

      ctx.ui.setWidget("voice-doctor", widgetLines);
      ctx.ui.setStatus("voice-doctor", "");

      if (allOk) {
        ctx.ui.notify(`🎉 /voice-doctor: all 5 checks passed — fully offline-capable`, "info");
      } else {
        ctx.ui.notify(`⚠️ /voice-doctor: ${passed}/5 passed. Failed: ${failed.join(", ")}`, "error");
      }
    },
  });

  // ─── Command: /voice-cache ─────────────────────────────────────────────
  pi.registerCommand("voice-cache", {
    description: `Show plugin cache location and contents at ${PLUGIN_CACHE_DIR}`,
    async handler(_args, ctx) {
      await ensureCacheDir();
      const cache = await isCacheComplete();
      const lines = [
        `Cache: \`${PLUGIN_CACHE_DIR.replace(homedir(), "~")}\``,
        "",
        ...Object.entries(cache.files).map(([name, info]) =>
          `  ${info.exists ? "✓" : "✗"} ${name}: ${
            info.exists ? (info.size / 1024 / 1024).toFixed(2) + " MB" : "missing"
          }`,
        ),
      ];
      const totalSize = Object.values(cache.files).reduce((s, f) => s + f.size, 0);
      lines.push("");
      lines.push(
        `Total: ${(totalSize / 1024 / 1024).toFixed(2)} MB — ${cache.complete ? "✓ complete" : "⚠️ incomplete"}`,
      );
      ctx.ui.setWidget("voice-cache", lines);
      ctx.ui.notify(
        `Cache: ${(totalSize / 1024 / 1024).toFixed(1)} MB, ${cache.complete ? "complete" : "incomplete"}`,
        cache.complete ? "info" : "warning",
      );
    },
  });
}
