import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { KokoroTTS } from "kokoro-js";
import { env as hfEnv } from "@huggingface/transformers";
import { spawn } from "node:child_process";
import { mkdtemp, rm, stat, access, readdir, writeFile, mkdir } from "node:fs/promises";
import { existsSync, createWriteStream } from "node:fs";
import { join } from "node:path";
import { tmpdir, homedir } from "node:os";
import { platform } from "node:process";
import { createRequire } from "node:module";
import https from "node:https";

const DEFAULT_VOICE = "am_michael";
const MODEL_ID = "onnx-community/Kokoro-82M-ONNX";

// ─── Self-managed cache location ─────────────────────────────────────────
// We use a stable, predictable location next to pi's agent cache.
// This way we own it, doctor can inspect it, and we don't depend on where
// @huggingface/transformers puts its defaults.
const PLUGIN_CACHE_DIR = join(homedir(), ".pi", "agent", "cache", "pi-voice-michael");
const MODEL_DIR = join(PLUGIN_CACHE_DIR, "model");
const ONNX_DIR = join(MODEL_DIR, "onnx");

// HF model file paths (relative to ONNX_DIR)
const MODEL_FILES = {
  "model_quantized.onnx": 90_000_000, // ~90 MB
  "config.json": 44,
  "tokenizer.json": 4608,
  "tokenizer_config.json": 113,
};

// Force @huggingface/transformers to use our cache directory BEFORE Kokoro
// instantiates its pipeline. This prevents the library from scattering files
// in node_modules-relative .cache directories.
hfEnv.cacheDir = PLUGIN_CACHE_DIR;
// Also disable remote downloads if user has opted in via env var
if (process.env.PI_VOICE_OFFLINE === "1") {
  hfEnv.allowRemoteModels = false;
  hfEnv.allowLocalModels = true;
}

// ─── Helpers ─────────────────────────────────────────────────────────────
function pickPlayer(): { cmd: string; args: (path: string) => string[] } | null {
  const p = platform;
  if (p === "darwin") return { cmd: "afplay", args: (f) => [f] };
  if (p === "win32") return {
    cmd: "powershell",
    args: (f) => ["-NoProfile", "-Command", `Add-Type -AssemblyName PresentationCore; (New-Object System.Media.SoundPlayer '${f.replace(/'/g, "''")}').PlaySync()`],
  };
  return {
    cmd: "sh",
    args: (f) => ["-c", `command -v paplay >/dev/null && paplay '${f}' || (command -v pw-play >/dev/null && pw-play '${f}' || aplay -q '${f}')`],
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
  const checks: { name: string; cmd: string }[] = platform === "darwin"
    ? [{ name: "afplay", cmd: "afplay" }]
    : platform === "win32"
    ? [{ name: "powershell", cmd: "powershell" }]
    : [
        { name: "paplay (PulseAudio)", cmd: "paplay" },
        { name: "pw-play (PipeWire)", cmd: "pw-play" },
        { name: "aplay (ALSA)", cmd: "aplay" },
        { name: "ffplay (ffmpeg fallback)", cmd: "ffplay" },
      ];
  return Promise.all(checks.map(async ({ name, cmd }) => {
    try {
      const res = await new Promise<{ status: number | null }>((resolve) => {
        const p = spawn("sh", ["-c", `command -v ${cmd}`], { stdio: "ignore" });
        p.on("exit", (code) => resolve({ status: code }));
        p.on("error", () => resolve({ status: -1 }));
      });
      return { name, available: res.status === 0 };
    } catch { return { name, available: false }; }
  }));
}

// ─── Self-managed model cache ────────────────────────────────────────────
async function ensureCacheDir(): Promise<void> {
  await mkdir(ONNX_DIR, { recursive: true });
  await mkdir(MODEL_DIR, { recursive: true });
}

async function isCacheComplete(): Promise<{ complete: boolean; files: Record<string, { exists: boolean; size: number }> }> {
  const files: Record<string, { exists: boolean; size: number }> = {};
  for (const [name, expected] of Object.entries(MODEL_FILES)) {
    const path = join(ONNX_DIR, name);
    if (existsSync(path)) {
      try {
        const s = await stat(path);
        files[name] = { exists: true, size: s.size };
      } catch { files[name] = { exists: false, size: 0 }; }
    } else {
      files[name] = { exists: false, size: expected };
    }
  }
  const complete = Object.entries(files).every(([name, f]) => f.exists && f.size >= MODEL_FILES[name as keyof typeof MODEL_FILES] * 0.5);
  return { complete, files };
}

// Download a single file from HF with progress reporting
function downloadFile(url: string, dest: string, onProgress?: (downloaded: number, total: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const follow = (url: string): void => {
      https.get(url, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          follow(res.headers.location);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
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

async function downloadModel(onProgress?: (msg: string, percent?: number) => void): Promise<void> {
  await ensureCacheDir();
  const { complete, files } = await isCacheComplete();
  if (complete) {
    onProgress?.("Model cache already complete", 100);
    return;
  }

  // HF resolves org/model paths: onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/<file>
  const baseUrl = (file: string) =>
    `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/${file}`;

  // JSON config files are at model root, not in onnx/ subfolder
  const configUrl = (file: string) =>
    `https://huggingface.co/${MODEL_ID}/resolve/main/${file}`;

  const downloads: { url: string; dest: string; label: string; weight: number }[] = [
    { url: baseUrl("model_quantized.onnx"), dest: join(ONNX_DIR, "model_quantized.onnx"), label: "ONNX model (~90 MB)", weight: 100 },
    { url: configUrl("config.json"), dest: join(ONNX_DIR, "config.json"), label: "config.json", weight: 0.01 },
    { url: configUrl("tokenizer.json"), dest: join(ONNX_DIR, "tokenizer.json"), label: "tokenizer.json", weight: 0.01 },
    { url: configUrl("tokenizer_config.json"), dest: join(ONNX_DIR, "tokenizer_config.json"), label: "tokenizer_config.json", weight: 0.001 },
  ];

  let totalSteps = 0;
  for (const d of downloads) {
    if (files[d.label === "ONNX model (~90 MB)" ? "model_quantized.onnx" : d.label]?.exists) continue;
    totalSteps++;
  }

  let step = 0;
  for (const d of downloads) {
    const filename = d.label === "ONNX model (~90 MB)" ? "model_quantized.onnx" : d.label;
    if (files[filename]?.exists) continue;

    step++;
    const pct = Math.round((step / totalSteps) * 100);
    onProgress?.(`Downloading ${d.label} (${step}/${totalSteps})...`, pct - 5);

    let lastReported = 0;
    await downloadFile(d.url, d.dest, (downloaded, total) => {
      const now = Math.floor((downloaded / (total || d.weight)) * 100);
      if (now > lastReported + 9) {
        onProgress?.(`Downloading ${d.label}: ${now}%`, pct - 5 + now / 100 * 5);
        lastReported = now;
      }
    });
  }
  onProgress?.("Model download complete", 100);
}

async function tryModelLoad(): Promise<{ ok: boolean; ms?: number; voices?: string[]; error?: string; onnxVersion?: string }> {
  const t0 = Date.now();
  try {
    const m = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
    const voices = (m as any).list_voices?.() ?? [];
    let onnxVersion = "unknown";
    try {
      const req = createRequire(import.meta.url ?? __filename);
      const ortPkg = req("onnxruntime-node/package.json");
      onnxVersion = ortPkg.version;
    } catch {}
    return { ok: true, ms: Date.now() - t0, voices, onnxVersion };
  } catch (err: any) {
    return { ok: false, error: err?.message ?? String(err) };
  }
}

async function tryEndToEnd(voice: string): Promise<{ ok: boolean; ms?: number; bytes?: number; error?: string }> {
  try {
    const tts = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
    const t0 = Date.now();
    const audio = await tts.generate("Voice test.", { voice } as any);
    const dir = await mkdtemp(join(tmpdir(), "pi-voice-doctor-"));
    const wavPath = join(dir, "test.wav");
    await audio.save(wavPath);
    const s = await stat(wavPath);
    await playWav(wavPath);
    rm(dir, { recursive: true, force: true }).catch(() => {});
    return { ok: true, ms: Date.now() - t0, bytes: s.size };
  } catch (err: any) {
    return { ok: false, error: err?.message ?? String(err) };
  }
}

async function tryPkgVersion(name: string): Promise<string | undefined> {
  try {
    const req = createRequire(import.meta.url ?? __filename);
    return req(`${name}/package.json`).version;
  } catch { return undefined; }
}

// ─── Plugin ──────────────────────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let tts: KokoroTTS | null = null;
  let ttsLoading: Promise<KokoroTTS> | null = null;

  async function ensureModelReady(onUpdate?: (msg: string, percent?: number) => void): Promise<void> {
    onUpdate?.("Preparing self-managed cache...", 0);
    await ensureCacheDir();
    const cache = await isCacheComplete();
    if (!cache.complete) {
      onUpdate?.("Model not in cache — downloading (one-time, ~90MB)...", 5);
      await downloadModel((msg, pct) => onUpdate?.(msg, pct));
      onUpdate?.("Model downloaded to plugin cache", 95);
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
    try {
      onUpdate?.("Initializing...", 5);
      const t0 = Date.now();
      const model = await ensureTTS(onUpdate);
      onUpdate?.(`Model ready in ${((Date.now() - t0) / 1000).toFixed(1)}s, generating speech...`, 70);

      const t1 = Date.now();
      const audio = await model.generate(text, { voice } as any);
      onUpdate?.(`Synthesized in ${((Date.now() - t1) / 1000).toFixed(1)}s, preparing playback...`, 90);

      const dir = await mkdtemp(join(tmpdir(), "pi-voice-"));
      const wavPath = join(dir, `speech-${Date.now()}.wav`);
      await audio.save(wavPath);
      onUpdate?.(`Playing through speakers...`, 95);

      try { await playWav(wavPath); }
      finally { rm(dir, { recursive: true, force: true }).catch(() => {}); }
      return { ok: true, voice, text, file: wavPath };
    } catch (err: any) {
      return { ok: false, voice, text, error: err?.message ?? String(err) };
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify(`🔊 pi-voice-michael loaded. Cache: ${PLUGIN_CACHE_DIR}. Run /voice-doctor to verify.`, "info");
    // Background pre-warm: ensure model is cached AND loaded
    (async () => {
      try {
        await ensureTTS();
      } catch {}
    })();
  });

  pi.on("session_shutdown", async () => {
    tts = null;
    ttsLoading = null;
  });

  // ─── Tool: voice_say_aloud ────────────────────────────────────────────
  pi.registerTool({
    name: "voice_say_aloud",
    label: "Speak Aloud (am_michael)",
    description:
      "Convert text to speech and play it aloud through the user's speakers using the offline am_michael voice (Kokoro ONNX, US English male). Self-managed cache at ~/.pi/agent/cache/pi-voice-michael/. Use this when the user explicitly asks you to speak, or when a verbal response is appropriate. Pass plain conversational text — no markdown, no code blocks, no URLs. First call may take ~30s for model download/init; subsequent calls are fast (~3s). If setup is broken, run /voice-doctor first.",
    parameters: Type.Object({
      text: Type.String({ description: "The text to speak aloud. Plain conversational English, no formatting. Keep it short (1–2 sentences ideal)." }),
      voice: Type.Optional(Type.String({ description: "Optional voice override. Defaults to am_michael. Other voices: am_fenrir, am_puck, bm_george, af_heart, af_bella, etc." })),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const { text, voice } = params as { text: string; voice?: string };
      const v = voice || DEFAULT_VOICE;
      const report = (msg: string, percent?: number) => {
        if (!onUpdate) return;
        try { onUpdate({ content: [{ type: "text", text: `🔊 ${msg}` }], details: { phase: msg, percent: percent ?? null } }); }
        catch {}
      };
      report("Starting TTS pipeline...", 0);
      const r = await speak(text, v, report);
      if (r.ok) {
        return {
          content: [{ type: "text", text: `🔊 Spoke aloud (${r.voice}): "${text}"` }],
          details: { ...r, percent: 100 },
        };
      }
      return {
        content: [{ type: "text", text: `❌ TTS playback failed: ${r.error}\nRun /voice-doctor to diagnose.` }],
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
      if (!trimmed) { ctx.ui.notify("Usage: /say <text> [voice]", "warning"); return; }
      const quotedMatch = trimmed.match(/^"([^"]+)"(?:\s+(\S+))?$/);
      let text: string, voice: string | undefined;
      if (quotedMatch) { text = quotedMatch[1]; voice = quotedMatch[2]; }
      else {
        const tokens = trimmed.split(/\s+/);
        if (tokens.length >= 2 && /^(am|af|bm|bf|jm)_/.test(tokens[tokens.length - 1])) voice = tokens.pop();
        text = tokens.join(" ");
      }
      ctx.ui.setStatus("pi-voice", "🔊 Initializing...");
      const r = await speak(text, voice || DEFAULT_VOICE, (msg, pct) => {
        ctx.ui.setStatus("pi-voice", `🔊 ${msg}${pct != null ? ` ${pct}%` : ""}`);
      });
      ctx.ui.setStatus("pi-voice", "");
      if (r.ok) ctx.ui.notify(`🔊 Spoke (${r.voice}): "${text}"`, "info");
      else ctx.ui.notify(`❌ TTS failed: ${r.error}`, "error");
    },
  });

  // ─── Command: /voice-doctor ───────────────────────────────────────────
  pi.registerCommand("voice-doctor", {
    description: "Diagnose pi-voice-michael setup — verifies full offline capability using self-managed cache",
    async handler(_args, ctx) {
      const lines: string[] = [];
      const checks: { ok: boolean; name: string }[] = [];

      lines.push("# pi-voice-michael Doctor Report");
      lines.push("");
      lines.push(`Platform: \`${platform} (${process.arch})\`, Node: \`${process.version}\``);
      lines.push(`Plugin cache: \`${PLUGIN_CACHE_DIR}\``);
      lines.push("");

      // ── 1. Audio player ────────────────────────────────────────────────
      lines.push("## 1. Audio Player");
      const players = await detectPlayers();
      const availPlayer = players.find((p) => p.available);
      for (const p of players) lines.push(`  ${p.available ? "✓" : "✗"} ${p.name}`);
      let playerOk = !!availPlayer;
      if (!playerOk) {
        lines.push("  ❌ **No audio player found.**");
        if (platform === "linux") {
          lines.push("    → Debian/Ubuntu: `apt install pulseaudio-utils` (paplay) or `alsa-utils` (aplay)");
          lines.push("    → Fedora: `dnf install pulseaudio-utils` or `alsa-utils`");
          lines.push("    → Arch: `pacman -S libpulse pipewire-pulse`");
        }
      } else {
        lines.push(`  ✓ Selected: **${availPlayer.name}**`);
      }
      checks.push({ ok: playerOk, name: "Audio player" });
      lines.push("");

      // ── 2. NPM dependencies ────────────────────────────────────────────
      lines.push("## 2. NPM Dependencies");
      const kokoroVer = await tryPkgVersion("kokoro-js");
      const transVer = await tryPkgVersion("@huggingface/transformers");
      const ortVer = await tryPkgVersion("onnxruntime-node");
      const piCodingVer = await tryPkgVersion("@earendil-works/pi-coding-agent");
      lines.push(`  ${kokoroVer ? "✓" : "❌"} kokoro-js \`${kokoroVer ?? "NOT INSTALLED"}\``);
      lines.push(`  ${transVer ? "✓" : "❌"} @huggingface/transformers \`${transVer ?? "NOT INSTALLED"}\``);
      lines.push(`  ${ortVer ? "✓" : "❌"} onnxruntime-node \`${ortVer ?? "NOT INSTALLED"}\``);
      lines.push(`  ${piCodingVer ? "✓" : "❌"} @earendil-works/pi-coding-agent \`${piCodingVer ?? "NOT INSTALLED"}\``);
      let depsOk = !!(kokoroVer && transVer && ortVer && piCodingVer);
      if (ortVer) {
        const minor = parseInt(ortVer.split(".")[1], 10);
        const ok = minor >= 20 && minor <= 21;
        if (!ok) {
          lines.push(`  ❌ **onnxruntime-node \`${ortVer}\` is INCOMPATIBLE with Kokoro q8 model.**`);
          lines.push(`     Fix: pin to **~1.21.0** in package.json, then reinstall`);
          depsOk = false;
        }
      }
      checks.push({ ok: depsOk, name: "Dependencies installed & compatible" });
      lines.push("");

      // ── 3. Self-managed cache ──────────────────────────────────────────
      lines.push("## 3. Plugin-Owned Model Cache");
      lines.push(`  Location: \`${PLUGIN_CACHE_DIR}\``);
      await ensureCacheDir();
      const cache = await isCacheComplete();
      let cacheOk = cache.complete;
      for (const [name, info] of Object.entries(cache.files)) {
        const sizeMB = (info.size / 1024 / 1024).toFixed(2);
        const expectedMB = (MODEL_FILES[name as keyof typeof MODEL_FILES] / 1024 / 1024).toFixed(2);
        lines.push(`  ${info.exists ? "✓" : "✗"} ${name}: ${info.exists ? sizeMB + " MB" : "missing"} (expected ${expectedMB} MB)`);
      }
      if (cacheOk) {
        lines.push(`  ✓ Cache is complete and self-contained — plugin will work fully offline`);
      } else {
        lines.push(`  ⚠️  Cache incomplete — plugin will download missing files on next speak call`);
      }
      checks.push({ ok: cacheOk, name: "Model cached for offline use" });
      lines.push("");

      // ── 4. Model load test ────────────────────────────────────────────
      lines.push("## 4. Model Load Test (live)");
      const loadResult = await tryModelLoad();
      let loadOk = false;
      if (loadResult.ok) {
        lines.push(`  ✓ Loaded in **${(loadResult.ms! / 1000).toFixed(2)}s** (onnxruntime-node ${loadResult.onnxVersion})`);
        if (loadResult.voices && loadResult.voices.length > 0) {
          lines.push(`  ✓ ${loadResult.voices.length} voices: ${loadResult.voices.slice(0, 8).join(", ")}${loadResult.voices.length > 8 ? "…" : ""}`);
          lines.push(`  ${loadResult.voices.includes(DEFAULT_VOICE) ? "✓" : "❌"} Default voice \`${DEFAULT_VOICE}\` available`);
          loadOk = loadResult.voices.includes(DEFAULT_VOICE);
        } else {
          loadOk = true;
        }
      } else {
        lines.push(`  ❌ Load failed: ${loadResult.error}`);
        lines.push(`     Common causes: onnxruntime-node version mismatch (see §2), corrupt model, native binding missing`);
      }
      checks.push({ ok: loadOk, name: "Model loads successfully" });
      lines.push("");

      // ── 5. End-to-end ──────────────────────────────────────────────────
      lines.push("## 5. End-to-End Test (synthesize + play)");
      if (!playerOk || !loadOk) {
        lines.push("  ⏭️  Skipped (prerequisites failed)");
        checks.push({ ok: false, name: "End-to-end playback" });
      } else {
        const e2e = await tryEndToEnd(DEFAULT_VOICE);
        if (e2e.ok) {
          lines.push(`  ✓ Synthesized + played in **${(e2e.ms! / 1000).toFixed(2)}s** (${(e2e.bytes! / 1024).toFixed(1)} KB WAV)`);
          checks.push({ ok: true, name: "End-to-end playback" });
        } else {
          lines.push(`  ❌ E2E failed: ${e2e.error}`);
          checks.push({ ok: false, name: "End-to-end playback" });
        }
      }
      lines.push("");

      // ── Summary ───────────────────────────────────────────────────────
      lines.push("## Summary");
      const passed = checks.filter((c) => c.ok).length;
      for (const c of checks) lines.push(`  ${c.ok ? "✓" : "❌"} ${c.name}`);
      lines.push("");
      const offlineCapable = checks.every((c) => c.ok);
      if (offlineCapable) {
        lines.push("🎉 **Plugin is 100% OFFLINE-CAPABLE.** Self-managed cache at:");
        lines.push(`   \`${PLUGIN_CACHE_DIR}\``);
      } else {
        const failed = checks.filter((c) => !c.ok).length;
        lines.push(`⚠️  **${failed} check(s) failed.** Fix above, then re-run /voice-doctor.`);
      }

      // Output strategy: split into multiple notify() calls (one per section)
      // so nothing gets truncated. The widget holds only a compact summary.
      const headline = offlineCapable
        ? `🎉 Voice TTS: 100% offline-capable (${passed}/${checks.length})`
        : `Voice TTS: ${passed}/${checks.length} checks passed — see issues below`;
      ctx.ui.notify(headline, offlineCapable ? "info" : "error");

      // Find the indices of the section headers (## N. ...) to slice lines
      const sectionHeaders: { idx: number; title: string }[] = [];
      for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(/^## (\d+\..*)/);
        if (m) sectionHeaders.push({ idx: i, title: m[1] });
      }
      // Push each section as its own notification
      for (let i = 0; i < sectionHeaders.length; i++) {
        const start = sectionHeaders[i].idx;
        const end = i + 1 < sectionHeaders.length ? sectionHeaders[i + 1].idx : lines.length - 1;
        // Also include "Summary" as the last section
        const block = lines.slice(start, end).join("\n");
        const failedInSection = block.includes("❌");
        ctx.ui.notify(block, failedInSection ? "error" : "info");
      }

      // Compact single-line widget — never truncated
      const summaryLines = [
        `pi-voice-michael: ${passed}/${checks.length} checks ✓`,
        ...checks.map((c) => `  ${c.ok ? "✓" : "❌"} ${c.name}`),
        offlineCapable ? "🎉 Fully offline-capable" : "⚠️  Issues reported above",
      ];
      ctx.ui.setWidget("voice-doctor", summaryLines);
    },
  });

  // ─── Command: /voice-cache ────────────────────────────────────────────
  pi.registerCommand("voice-cache", {
    description: `Show plugin cache location and contents at ${PLUGIN_CACHE_DIR}`,
    async handler(_args, ctx) {
      await ensureCacheDir();
      const cache = await isCacheComplete();
      const lines = [
        `Cache directory: \`${PLUGIN_CACHE_DIR}\``,
        "",
        ...Object.entries(cache.files).map(([name, info]) =>
          `  ${info.exists ? "✓" : "✗"} ${name}: ${info.exists ? (info.size / 1024 / 1024).toFixed(2) + " MB" : "missing"}`
        ),
      ];
      const totalSize = Object.values(cache.files).reduce((s, f) => s + f.size, 0);
      lines.push("");
      lines.push(`Total: ${(totalSize / 1024 / 1024).toFixed(2)} MB across ${Object.keys(cache.files).length} files`);
      lines.push(`Cache complete: ${cache.complete ? "✓ yes" : "✗ no"}`);
      ctx.ui.setWidget("voice-cache", lines);
      ctx.ui.notify(`Cache: ${(totalSize / 1024 / 1024).toFixed(1)} MB, complete: ${cache.complete}`, cache.complete ? "info" : "warning");
    },
  });
}