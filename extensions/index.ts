import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile, readFile, stat, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { tmpdir, homedir } from "node:os";
import { platform } from "node:process";
import { loadVoiceStyle, loadTextToSpeech, writeWavFile } from "../src/supertonic-helper.js";

// ─── Self-managed plugin cache ─────────────────────────────────────────
//
// ~/.pi/voice/supertonic/
//   ├── config.json                       # user settings (default voice + lang)
//   ├── onnx/                              # ~380MB ONNX model files
//   │   ├── duration_predictor.onnx
//   │   ├── text_encoder.onnx
//   │   ├── vector_estimator.onnx
//   │   ├── vocoder.onnx
//   │   ├── tts.json
//   │   └── unicode_indexer.json
//   └── voice_styles/                     # 10 voice style presets
//       ├── M1.json ... M5.json            # male voices
//       └── F1.json ... F5.json            # female voices
const PLUGIN_CACHE_DIR = join(homedir(), ".pi", "voice", "supertonic");
const ONNX_DIR = join(PLUGIN_CACHE_DIR, "onnx");
const VOICE_STYLES_DIR = join(PLUGIN_CACHE_DIR, "voice_styles");

const DEFAULT_VOICE = "M1";
const DEFAULT_LANG = "de";
const DEFAULT_SPEED = 1.05;
const DEFAULT_TOTAL_STEP = 5;

const MODEL_FILES = [
  "duration_predictor.onnx",
  "text_encoder.onnx",
  "vector_estimator.onnx",
  "vocoder.onnx",
  "tts.json",
  "unicode_indexer.json",
];

const AVAILABLE_VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"];
const AVAILABLE_LANGS = ["en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"];

// ─── Settings ────────────────────────────────────────────────────────────
const SETTINGS_FILE = process.env.PI_VOICE_SETTINGS ?? join(PLUGIN_CACHE_DIR, "config.json");

interface VoiceSettings {
  voice: string;
  lang: string;
  speed: number;
  totalStep: number;
}

async function loadSettings(): Promise<VoiceSettings> {
  try {
    const raw = await readFile(SETTINGS_FILE, "utf8");
    const p = JSON.parse(raw);
    return {
      voice: typeof p.voice === "string" ? p.voice : DEFAULT_VOICE,
      lang: typeof p.lang === "string" ? p.lang : DEFAULT_LANG,
      speed: typeof p.speed === "number" ? p.speed : DEFAULT_SPEED,
      totalStep: typeof p.totalStep === "number" ? p.totalStep : DEFAULT_TOTAL_STEP,
    };
  } catch {
    return { voice: DEFAULT_VOICE, lang: DEFAULT_LANG, speed: DEFAULT_SPEED, totalStep: DEFAULT_TOTAL_STEP };
  }
}

async function saveSettings(s: VoiceSettings): Promise<void> {
  await mkdir(dirname(SETTINGS_FILE), { recursive: true });
  await writeFile(SETTINGS_FILE, JSON.stringify(s, null, 2), "utf8");
}

// ─── Cache management ────────────────────────────────────────────────────
async function ensureCacheDir(): Promise<void> {
  await mkdir(PLUGIN_CACHE_DIR, { recursive: true });
  await mkdir(ONNX_DIR, { recursive: true });
  await mkdir(VOICE_STYLES_DIR, { recursive: true });
}

interface CacheStatus {
  complete: boolean;
  files: Record<string, { exists: boolean; size: number }>;
}

async function isCacheComplete(): Promise<CacheStatus> {
  const files: Record<string, { exists: boolean; size: number }> = {};
  for (const name of MODEL_FILES) {
    const p = join(ONNX_DIR, name);
    if (existsSync(p)) {
      try {
        const s = await stat(p);
        files[name] = { exists: true, size: s.size };
      } catch {
        files[name] = { exists: false, size: 0 };
      }
    } else {
      files[name] = { exists: false, size: 0 };
    }
  }
  const complete = MODEL_FILES.every((f) => files[f]?.exists && files[f].size > 1000);
  return { complete, files };
}

function missingModelError(c?: CacheStatus): string {
  const base = `Supertonic 3 model not found or incomplete in plugin cache.\n` +
    `Required location: ${ONNX_DIR}/\n`;
  if (c) {
    const missing = MODEL_FILES.filter((f) => !c.files[f]?.exists).map((f) => `  - ${f}`);
    const broken = MODEL_FILES.filter((f) => c.files[f]?.exists && c.files[f]!.size < 1000).map((f) => `  - ${f} (${c.files[f]!.size} bytes — likely download error)`);
    return base +
      (missing.length ? `\nMissing:\n${missing.join("\n")}` : "") +
      (broken.length ? `\nCorrupt:\n${broken.join("\n")}` : "") +
      `\n\nDownload from: https://huggingface.co/Supertone/supertonic-3/tree/main/onnx\n` +
      `Or set PI_VOICE_ONLINE=1 to allow auto-download.`;
  }
  return base +
    `Required files: ${MODEL_FILES.join(", ")}\n\n` +
    `Download from: https://huggingface.co/Supertone/supertonic-3/tree/main/onnx\n` +
    `Or set PI_VOICE_ONLINE=1 to allow auto-download.`;
}

// ─── Audio playback ───────────────────────────────────────────────────────
function pickPlayer(): { cmd: string; args: (path: string) => string[] } | null {
  const p = platform;
  if (p === "darwin") return { cmd: "afplay", args: (f) => [f] };
  if (p === "win32") {
    return {
      cmd: "powershell",
      args: (f) => ["-NoProfile", "-Command", `Add-Type -AssemblyName PresentationCore; (New-Object System.Media.SoundPlayer '${f.replace(/'/g, "''")}').PlaySync()`],
    };
  }
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
  const checks = platform === "darwin"
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
    } catch {
      return { name, available: false };
    }
  }));
}

// ─── Plugin ──────────────────────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let settings: VoiceSettings = { voice: DEFAULT_VOICE, lang: DEFAULT_LANG, speed: DEFAULT_SPEED, totalStep: DEFAULT_TOTAL_STEP };

  async function speak(
    text: string,
    onUpdate?: (msg: string, percent?: number) => void,
  ): Promise<{ ok: boolean; voice: string; lang: string; text: string; error?: string; durationSec?: number }> {
    const dir = await mkdtemp(join(tmpdir(), "pi-voice-"));
    const wavPath = join(dir, `speech-${Date.now()}.wav`);
    try {
      onUpdate?.("Initializing Supertonic…", 5);
      await ensureCacheDir();
      const cache = await isCacheComplete();
      if (!cache.complete) {
        throw new Error(missingModelError(cache));
      }

      onUpdate?.("Loading model (warm: <500ms)…", 30);
      const t0 = Date.now();
      const tts = await loadTextToSpeech(ONNX_DIR, false);
      const voiceStyle = await loadVoiceStyle([join(VOICE_STYLES_DIR, `${settings.voice}.json`)]);
      onUpdate?.(`Model ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`, 60);

      onUpdate?.("Synthesizing…", 75);
      const t1 = Date.now();
      // TextToSpeech.call() handles text chunking internally for long inputs
      // (maxLen 300 chars per chunk, or 120 for ja/ko)
      const { wav, duration } = await tts.call(text, settings.lang, voiceStyle, settings.totalStep, settings.speed);
      const genMs = Date.now() - t1;

      await writeWavFile(wavPath, wav, tts.sampleRate);
      const audioSec = (duration[0] !== undefined && duration[0] !== null) ? duration[0] : wav.length / tts.sampleRate;
      onUpdate?.(`Synthesized ${audioSec.toFixed(1)}s audio in ${(genMs / 1000).toFixed(1)}s, playing…`, 95);

      try {
        await playWav(wavPath);
      } catch (err) {
        onUpdate?.(`Playback failed: ${(err as Error).message}`, 100);
      }
      onUpdate?.("Done", 100);
      return { ok: true, voice: settings.voice, lang: settings.lang, text, durationSec: audioSec };
    } catch (err) {
      return { ok: false, voice: settings.voice, lang: settings.lang, text, error: (err as Error).message };
    } finally {
      void rm(dir, { recursive: true, force: true }).catch(() => {});
    }
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────
  pi.on("session_start", async (_event, ctx) => {
    settings = await loadSettings();
    ctx.ui.notify(
      `🔊 pi-voice-michael (Supertonic 3) loaded. Voice: ${settings.voice}, Lang: ${settings.lang}. Cache: ${PLUGIN_CACHE_DIR.replace(homedir(), "~")}`,
      "info",
    );
  });

  // ─── Tool: voice_say_aloud ────────────────────────────────────────────
  pi.registerTool({
    name: "voice_say_aloud",
    label: "Speak Aloud (Supertonic)",
    description:
      "Convert text to speech and play it aloud through the user's speakers using the offline Supertonic 3 ONNX engine (44.1kHz, 99M params, 31 languages). Configure voice via /voice-set and language via /voice-lang. Pass plain conversational text — no markdown, no code blocks. Long texts are automatically chunked internally. First call ~500ms model load, then ~2-4s per sentence.",
    parameters: Type.Object({
      text: Type.String({ description: "Plain conversational text to speak. No formatting, no URLs." }),
    }),
    async execute(_id, params, _signal, onUpdate, _ctx) {
      const { text } = params as { text: string };
      const report = (msg: string, percent?: number) => {
        if (!onUpdate) return;
        try { onUpdate({ content: [{ type: "text", text: `🔊 ${msg}` }], details: { phase: msg, percent: percent ?? null } }); }
        catch { /* onUpdate may not be available */ }
      };
      report("Starting TTS pipeline...", 0);
      const r = await speak(text, report);
      if (r.ok) {
        return {
          content: [{ type: "text", text: `🔊 Spoke (${r.voice}, ${r.lang}, ${r.durationSec?.toFixed(1)}s audio): "${text.slice(0, 60)}${text.length > 60 ? "..." : ""}"` }],
          details: r,
        };
      }
      return {
        content: [{ type: "text", text: `❌ TTS failed: ${r.error}\nRun /voice-doctor to diagnose.` }],
        details: r,
        isError: true,
      };
    },
  });

  // ─── Command: /say ────────────────────────────────────────────────────
  pi.registerCommand("say", {
    description: "Speak text aloud using the configured Supertonic voice. Usage: /say <text>",
    async handler(args, ctx) {
      const text = args.trim();
      if (!text) {
        ctx.ui.notify(`Usage: /say <text>  (voice: ${settings.voice}, lang: ${settings.lang})`, "warning");
        return;
      }
      ctx.ui.setStatus("pi-voice", "🔊 Loading Supertonic…");
      const r = await speak(text, (msg, pct) => {
        ctx.ui.setStatus("pi-voice", `🔊 ${msg}${pct != null ? ` ${pct}%` : ""}`);
      });
      ctx.ui.setStatus("pi-voice", "");
      if (r.ok) ctx.ui.notify(`🔊 Spoke (${r.durationSec?.toFixed(1)}s audio): "${text.slice(0, 80)}${text.length > 80 ? "..." : ""}"`, "info");
      else ctx.ui.notify(`❌ TTS failed: ${r.error}`, "error");
    },
  });

  // ─── Command: /voice-set ──────────────────────────────────────────────
  pi.registerCommand("voice-set", {
    description: `Set the voice style. Available: ${AVAILABLE_VOICES.join(", ")}`,
    async handler(args, ctx) {
      const v = args.trim().toUpperCase();
      if (!AVAILABLE_VOICES.includes(v)) {
        ctx.ui.notify(`❌ Unknown voice: ${v}\nAvailable: ${AVAILABLE_VOICES.join(", ")}`, "error");
        return;
      }
      settings.voice = v;
      await saveSettings(settings);
      ctx.ui.notify(`✓ Voice set to **${v}** (saved to ${SETTINGS_FILE.replace(homedir(), "~")})`, "info");
    },
  });

  // ─── Command: /voice-lang ─────────────────────────────────────────────
  pi.registerCommand("voice-lang", {
    description: `Set the language code. Available: ${AVAILABLE_LANGS.join(", ")}`,
    async handler(args, ctx) {
      const l = args.trim().toLowerCase();
      if (!AVAILABLE_LANGS.includes(l)) {
        ctx.ui.notify(`❌ Unknown lang: ${l}\nAvailable: ${AVAILABLE_LANGS.join(", ")}`, "error");
        return;
      }
      settings.lang = l;
      await saveSettings(settings);
      ctx.ui.notify(`✓ Language set to **${l}**`, "info");
    },
  });

  // ─── Command: /voice-list ─────────────────────────────────────────────
  pi.registerCommand("voice-list", {
    description: "List available voices and languages",
    async handler(_args, ctx) {
      const lines = [
        `Current: voice=${settings.voice}, lang=${settings.lang}, speed=${settings.speed}, totalStep=${settings.totalStep}`,
        "",
        `Voices (${AVAILABLE_VOICES.length}):`,
        `  Male:   ${AVAILABLE_VOICES.filter(v => v.startsWith("M")).join(", ")}`,
        `  Female: ${AVAILABLE_VOICES.filter(v => v.startsWith("F")).join(", ")}`,
        "",
        `Languages (${AVAILABLE_LANGS.length}):`,
        `  ${AVAILABLE_LANGS.join(", ")}`,
      ];
      ctx.ui.setWidget("voice-list", lines);
      ctx.ui.notify(`${AVAILABLE_VOICES.length} voices × ${AVAILABLE_LANGS.length} languages available`, "info");
    },
  });

  // ─── Command: /voice-cache ────────────────────────────────────────────
  pi.registerCommand("voice-cache", {
    description: `Show Supertonic cache status at ${PLUGIN_CACHE_DIR}`,
    async handler(_args, ctx) {
      await ensureCacheDir();
      const cache = await isCacheComplete();
      const totalSize = Object.values(cache.files).reduce((s, f) => s + f.size, 0);
      const lines = [
        `Settings: ${SETTINGS_FILE.replace(homedir(), "~")}`,
        `  voice=${settings.voice}, lang=${settings.lang}, speed=${settings.speed}, totalStep=${settings.totalStep}`,
        "",
        `Cache: ${PLUGIN_CACHE_DIR.replace(homedir(), "~")}`,
        ...MODEL_FILES.map((f) => `  ${cache.files[f]?.exists ? "✓" : "✗"} onnx/${f}: ${cache.files[f]?.exists ? `${(cache.files[f]!.size / 1024 / 1024).toFixed(1)} MB` : "MISSING"}`),
        "",
        `Total: ${(totalSize / 1024 / 1024).toFixed(1)} MB`,
        cache.complete ? "✓ Cache complete — ready for offline use" : "⚠️  Cache incomplete",
      ];
      ctx.ui.setWidget("voice-cache", lines);
      ctx.ui.notify(`Cache: ${(totalSize / 1024 / 1024).toFixed(1)} MB — ${cache.complete ? "complete" : "incomplete"}`, cache.complete ? "info" : "warning");
    },
  });

  // ─── Command: /voice-doctor ───────────────────────────────────────────
  pi.registerCommand("voice-doctor", {
    description: "Diagnose Supertonic setup — verifies offline capability",
    async handler(_args, ctx) {
      ctx.ui.setStatus("voice-doctor", "Running…");
      const checks: { ok: boolean; name: string; detail: string }[] = [];

      // 1. Audio player
      const players = await detectPlayers();
      const avail = players.find((p) => p.available);
      checks.push({ ok: !!avail, name: "Audio player", detail: avail ? avail.name : "none found" });
      ctx.ui.notify(`1/5 ${avail ? "✓" : "❌"} Audio: ${avail?.name ?? "none"}`, avail ? "info" : "error");

      // 2. Cache
      const cache = await isCacheComplete();
      const totalMB = Object.values(cache.files).reduce((s, f) => s + f.size, 0) / 1024 / 1024;
      checks.push({ ok: cache.complete, name: "Supertonic model cache", detail: `${totalMB.toFixed(0)} MB at ${PLUGIN_CACHE_DIR}` });
      ctx.ui.notify(`2/5 ${cache.complete ? "✓" : "❌"} Cache: ${totalMB.toFixed(0)} MB`, cache.complete ? "info" : "error");

      // 3. Settings
      checks.push({ ok: true, name: "Settings", detail: settings.voice + "/" + settings.lang });
      ctx.ui.notify(`3/5 ✓ Settings: ${settings.voice}/${settings.lang}`, "info");

      // 4. Model load
      let loadOk = false;
      let loadDetail = "skipped";
      if (cache.complete) {
        try {
          const t0 = Date.now();
          await loadTextToSpeech(ONNX_DIR, false);
          loadDetail = `loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`;
          loadOk = true;
        } catch (err) {
          loadDetail = `failed: ${(err as Error).message}`;
        }
      } else {
        loadDetail = "skipped (no model)";
      }
      checks.push({ ok: loadOk, name: "Model load", detail: loadDetail });
      ctx.ui.notify(`4/5 ${loadOk ? "✓" : "❌"} Model: ${loadDetail}`, loadOk ? "info" : "error");

      // 5. End-to-end
      let e2eOk = false;
      if (loadOk && avail) {
        try {
          const r = await speak("Test.", () => {});
          e2eOk = r.ok;
          if (r.ok) ctx.ui.notify(`5/5 ✓ E2E: ${r.durationSec?.toFixed(1)}s audio OK`, "info");
          else ctx.ui.notify(`5/5 ❌ E2E: ${r.error}`, "error");
        } catch (err) {
          ctx.ui.notify(`5/5 ❌ E2E: ${(err as Error).message}`, "error");
        }
      } else {
        ctx.ui.notify("5/5 ⏭️  E2E: skipped (prereqs failed)", "info");
      }
      checks.push({ ok: e2eOk, name: "End-to-end", detail: e2eOk ? "OK" : "skipped/failed" });

      ctx.ui.setStatus("voice-doctor", "");
      const passed = checks.filter((c) => c.ok).length;
      const widgetLines = [
        `Supertonic 3 Doctor: ${passed}/5 checks ✓`,
        ...checks.map((c) => `  ${c.ok ? "✓" : "❌"} ${c.name}: ${c.detail}`),
        passed === 5 ? "\n🎉 Fully offline-capable" : `\n⚠️  ${5 - passed} issue(s) above`,
      ];
      ctx.ui.setWidget("voice-doctor", widgetLines);
      ctx.ui.notify(
        passed === 5 ? `🎉 Voice TTS: 100% offline-capable` : `Voice TTS: ${passed}/5 checks passed — see issues above`,
        passed === 5 ? "info" : "warning",
      );
    },
  });
}