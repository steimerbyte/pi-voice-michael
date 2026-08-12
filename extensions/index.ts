import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { KokoroTTS } from "kokoro-js";
import { spawn } from "node:child_process";
import { mkdtemp, rm, stat, access, readdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir, homedir } from "node:os";
import { platform } from "node:process";

const DEFAULT_VOICE = "am_michael";
const MODEL_ID = "onnx-community/Kokoro-82M-ONNX";

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

async function detectPlayers(): Promise<{ name: string; available: boolean; path?: string }[]> {
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
      return { name, available: res.status === 0, path: cmd };
    } catch {
      return { name, available: false };
    }
  }));
}

// All known @huggingface/transformers cache locations — kokoro-js may store
// its cache next to its own node_modules or in the user's HF home.
async function findHFCacheForKokoro(): Promise<{ dir: string; exists: boolean; modelFiles?: string[]; sizeBytes?: number }[]> {
  const candidates = [
    // Standard HF cache layout (downloads from HF Hub)
    join(homedir(), ".cache", "huggingface", "hub", "models--onnx-community--Kokoro-82M-ONNX"),
    // @huggingface/transformers "transformers.js" cache layout
    join(homedir(), ".cache", "huggingface", "models--onnx-community--Kokoro-82M-ONNX"),
    // Maybe-process-local? kokoro-js uses node_modules-relative path by default
    process.cwd(),
  ];
  const results: { dir: string; exists: boolean; modelFiles?: string[]; sizeBytes?: number }[] = [];
  for (const dir of candidates) {
    const exists = existsSync(dir);
    let modelFiles: string[] | undefined;
    let sizeBytes: number | undefined;
    if (exists) {
      try {
        const files = await readdir(dir).catch(() => []);
        modelFiles = files.slice(0, 10);
        // Sum sizes
        let total = 0;
        for (const f of files) {
          try { total += (await stat(join(dir, f))).size; } catch {}
        }
        sizeBytes = total;
      } catch {}
    }
    results.push({ dir, exists, modelFiles, sizeBytes });
  }
  return results;
}

// Try to actually load the model in isolation. This catches the "Protobuf parsing failed"
// class of bugs that don't surface until runtime.
async function tryModelLoad(): Promise<{ ok: boolean; ms?: number; voices?: string[]; error?: string; onnxVersion?: string }> {
  const t0 = Date.now();
  try {
    const m = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" });
    const voices = (m as any).list_voices?.() ?? [];
    let onnxVersion = "unknown";
    try {
      const ortPkg = await import("onnxruntime-node/package.json", { with: { type: "json" } } as any);
      onnxVersion = (ortPkg as any).default.version;
    } catch {}
    return { ok: true, ms: Date.now() - t0, voices, onnxVersion };
  } catch (err: any) {
    return { ok: false, error: err?.message ?? String(err) };
  }
}

// Try to actually synthesize a test utterance and play it
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

// Read package.json version of a dep without throwing
async function tryPkgVersion(name: string): Promise<string | undefined> {
  try {
    const mod: any = await import(`${name}/package.json`, { with: { type: "json" } } as any);
    return mod.default?.version;
  } catch {
    return undefined;
  }
}

// ─── Plugin ──────────────────────────────────────────────────────────────
export default function (pi: ExtensionAPI) {
  let tts: KokoroTTS | null = null;
  let ttsLoading: Promise<KokoroTTS> | null = null;

  async function ensureTTS(): Promise<KokoroTTS> {
    if (tts) return tts;
    if (!ttsLoading) {
      ttsLoading = KokoroTTS.from_pretrained(MODEL_ID, { dtype: "q8", device: "cpu" })
        .then((m) => { tts = m; return m; });
    }
    return ttsLoading;
  }

  async function speak(
    text: string,
    voice: string,
    onUpdate?: (msg: string, percent?: number) => void,
  ): Promise<{ ok: boolean; voice: string; text: string; file?: string; error?: string }> {
    try {
      onUpdate?.("Initializing ONNX runtime...", 5);
      const t0 = Date.now();
      const model = await ensureTTS();
      onUpdate?.(`Model loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s, generating speech...`, 70);

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
    ctx.ui.notify("🔊 pi-voice-michael loaded (Kokoro am_michael). Run /voice-doctor to verify offline setup.", "info");
    ensureTTS().catch(() => { /* surfaced on first tool call */ });
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
      "Convert text to speech and play it aloud through the user's speakers using the offline am_michael voice (Kokoro ONNX, US English male). Use this when the user explicitly asks you to speak, or when a verbal response is appropriate. Pass plain conversational text — no markdown, no code blocks, no URLs. First call may take ~30s for model initialization; subsequent calls are fast (~3s). If setup is broken, run /voice-doctor first.",
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
        catch { /* onUpdate may not be available */ }
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
      ctx.ui.setStatus("pi-voice", "🔊 Loading Kokoro TTS...");
      const r = await speak(text, voice || DEFAULT_VOICE, (msg, pct) => {
        ctx.ui.setStatus("pi-voice", `🔊 ${msg}${pct != null ? ` ${pct}%` : ""}`);
      });
      ctx.ui.setStatus("pi-voice", "");
      if (r.ok) ctx.ui.notify(`🔊 Spoke (${r.voice}): "${text}"`, "info");
      else ctx.ui.notify(`❌ TTS failed: ${r.error}`, "error");
    },
  });

  // ─── Command: /voice-doctor ───────────────────────────────────────────
  // Verifies that voice_say_aloud will work 100% OFFLINE after installation
  // is complete. Runs an end-to-end synthesis test to catch runtime issues.
  pi.registerCommand("voice-doctor", {
    description: "Diagnose pi-voice-michael setup for full offline operation",
    async handler(_args, ctx) {
      const lines: string[] = [];
      const checks: { ok: boolean; name: string; detail?: string }[] = [];

      lines.push("# pi-voice-michael Doctor Report");
      lines.push("");
      lines.push(`Platform: \`${platform} (${process.arch})\`, Node: \`${process.version}\``);
      lines.push("");

      // ── 1. Audio player ────────────────────────────────────────────────
      lines.push("## 1. Audio Player");
      const players = await detectPlayers();
      const availPlayer = players.find((p) => p.available);
      for (const p of players) lines.push(`  ${p.available ? "✓" : "✗"} ${p.name}`);
      let playerOk = false;
      if (!availPlayer) {
        lines.push("  ❌ **No audio player found.** Install one:");
        if (platform === "linux") {
          lines.push("    - Debian/Ubuntu: `apt install pulseaudio-utils` (paplay) or `alsa-utils` (aplay)");
          lines.push("    - Fedora: `dnf install pulseaudio-utils` or `alsa-utils`");
          lines.push("    - Arch: `pacman -S libpulse pipewire-pulse`");
        } else if (platform === "darwin") {
          lines.push("    - afplay is built-in; this should never fail.");
        }
      } else {
        lines.push(`  ✓ Selected: **${availPlayer.name}**`);
        playerOk = true;
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

      // onnxruntime-node version compatibility check
      if (ortVer) {
        const major = parseInt(ortVer.split(".")[0], 10);
        const minor = parseInt(ortVer.split(".")[1], 10);
        const ok = (major === 1 && minor >= 20 && minor <= 21);
        if (!ok) {
          lines.push(`  ❌ **onnxruntime-node \`${ortVer}\` is INCOMPATIBLE with Kokoro q8 model.**`);
          lines.push(`     Known broken range: ≥ 1.22 (Protobuf parsing failed)`);
          lines.push(`     Fix: pin to **~1.21.0** in package.json, then reinstall`);
          depsOk = false;
        }
      }
      checks.push({ ok: depsOk, name: "Dependencies installed & compatible" });
      lines.push("");

      // ── 3. Model cache ─────────────────────────────────────────────────
      lines.push("## 3. ONNX Model Cache (offline requirement)");
      const cacheResults = await findHFCacheForKokoro();
      let cacheOk = false;
      let cacheTotal = 0;
      for (const c of cacheResults) {
        if (c.exists) {
          const mb = c.sizeBytes ? (c.sizeBytes / 1024 / 1024).toFixed(1) : "?";
          lines.push(`  ✓ \`${c.dir}\` exists (${mb} MB)`);
          cacheOk = true;
          cacheTotal += c.sizeBytes ?? 0;
        } else {
          lines.push(`  ✗ \`${c.dir}\` not found`);
        }
      }
      if (!cacheOk) {
        lines.push("  ❌ **Model not cached.** Plugin would need internet on first use.");
        lines.push("     Run once online, or manually download model_quantized.onnx from:");
        lines.push("     https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx");
        lines.push("     to ~/.cache/huggingface/hub/models--onnx-community--Kokoro-82M-ONNX/snapshots/<hash>/onnx/");
      } else if (cacheTotal < 80 * 1024 * 1024) {
        lines.push(`  ⚠️  Cache only ${(cacheTotal / 1024 / 1024).toFixed(1)} MB — expected ~90 MB. Model may be incomplete.`);
        cacheOk = false;
      }
      checks.push({ ok: cacheOk, name: "Model cached for offline use" });
      lines.push("");

      // ── 4. Model load test (catches runtime incompatibilities) ────────
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
          lines.push("  ⚠️  Could not enumerate voices (list_voices missing?)");
          loadOk = true; // load succeeded, assume OK
        }
      } else {
        lines.push(`  ❌ Load failed: ${loadResult.error}`);
        lines.push("     This is the most common failure mode. Common causes:");
        lines.push("     - onnxruntime-node version mismatch (see section 2)");
        lines.push("     - model file corrupted (delete cache, re-download)");
        lines.push("     - native binding missing for your platform");
      }
      checks.push({ ok: loadOk, name: "Model loads successfully" });
      lines.push("");

      // ── 5. End-to-end synthesis + playback ─────────────────────────────
      lines.push("## 5. End-to-End Test (synthesize + play)");
      if (!playerOk || !loadOk) {
        lines.push("  ⏭️  Skipped (prerequisites failed)");
        checks.push({ ok: false, name: "End-to-end playback" });
      } else {
        const e2e = await tryEndToEnd(DEFAULT_VOICE);
        if (e2e.ok) {
          lines.push(`  ✓ Synthesized + played in **${(e2e.ms! / 1000).toFixed(2)}s** (${(e2e.bytes! / 1024).toFixed(1)} KB WAV)`);
          lines.push(`  ✓ Audio output verified — plugin is **fully functional**`);
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
      const total = checks.length;
      for (const c of checks) lines.push(`  ${c.ok ? "✓" : "❌"} ${c.name}`);
      lines.push("");
      const offlineCapable = checks.every((c) => c.ok);
      if (offlineCapable) {
        lines.push("🎉 **Plugin is 100% OFFLINE-CAPABLE.** voice_say_aloud will work without internet.");
      } else {
        const failed = checks.filter((c) => !c.ok).length;
        lines.push(`⚠️  **${failed} check(s) failed.** Plugin may require internet on next run, or fail entirely.`);
      }

      const report = lines.join("\n");
      const headline = offlineCapable
        ? "🎉 Voice TTS: 100% offline-capable"
        : `Voice TTS: ${passed}/${total} checks passed`;
      ctx.ui.notify(headline, offlineCapable ? "info" : "error");
      ctx.ui.setWidget("voice-doctor", lines);
    },
  });
}