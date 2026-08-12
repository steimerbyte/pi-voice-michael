import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { KokoroTTS } from "kokoro-js";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

const DEFAULT_VOICE = "am_michael";

interface SpeakResult {
  ok: boolean;
  voice: string;
  text: string;
  file?: string;
  error?: string;
}

type AgentToolUpdateCallback<T = unknown> = (update: { content?: string; details?: T; isError?: boolean }) => void;

function pickPlayer(): { cmd: string; args: (path: string) => string[] } | null {
  const p = process.platform;
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
  if (!player) return Promise.reject(new Error(`No audio player for platform ${process.platform}`));
  return new Promise((resolve, reject) => {
    const proc = spawn(player.cmd, player.args(wavPath), { stdio: "ignore" });
    proc.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`audio player exited ${code}`))));
    proc.on("error", reject);
  });
}

export default function (pi: ExtensionAPI) {
  let tts: KokoroTTS | null = null;
  let ttsLoading: Promise<KokoroTTS> | null = null;

  async function ensureTTS(onUpdate?: AgentToolUpdateCallback): Promise<KokoroTTS> {
    if (tts) return tts;
    if (!ttsLoading) {
      onUpdate?.({ content: "⏳ Loading Kokoro TTS model (~90MB, first run only)..." });
      ttsLoading = KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-ONNX", {
        dtype: "q8",
        device: "cpu",
      }).then((m) => {
        onUpdate?.({ content: "✅ Kokoro model loaded, generating speech..." });
        tts = m;
        return m;
      });
    } else {
      onUpdate?.({ content: "⏳ Model already loading, please wait..." });
    }
    return ttsLoading;
  }

  async function speak(text: string, voice: string = DEFAULT_VOICE, onUpdate?: AgentToolUpdateCallback): Promise<SpeakResult> {
    try {
      onUpdate?.({ content: `🎤 Synthesizing: "${text.substring(0, 60)}${text.length > 60 ? "..." : ""}"` });
      const model = await ensureTTS(onUpdate);
      const audio = await model.generate(text, { voice } as any);
      onUpdate?.({ content: "💾 Saving audio file..." });
      const dir = await mkdtemp(join(tmpdir(), "pi-voice-"));
      const wavPath = join(dir, `speech-${Date.now()}.wav`);
      await audio.save(wavPath);
      onUpdate?.({ content: `🔊 Playing: ${wavPath}` });
      try {
        await playWav(wavPath);
      } finally {
        rm(dir, { recursive: true, force: true }).catch(() => {});
      }
      return { ok: true, voice, text, file: wavPath };
    } catch (err: any) {
      return { ok: false, voice, text, error: err?.message ?? String(err) };
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify("🔊 pi-voice-michael ready (Kokoro am_michael)", "info");
    // Pre-warm the model in background so first real call is faster
    ensureTTS().catch(() => {});
  });

  pi.on("session_shutdown", async () => {
    tts = null;
    ttsLoading = null;
  });

  pi.registerTool({
    name: "voice_say_aloud",
    label: "Speak Aloud (am_michael)",
    description:
      "Convert text to speech and play it aloud through the user's speakers using the offline am_michael voice (Kokoro ONNX, US English male). Use this when the user explicitly asks you to speak, or when a verbal response is appropriate. Pass plain conversational text — no markdown, no code blocks, no URLs.",
    parameters: Type.Object({
      text: Type.String({
        description:
          "The text to speak aloud. Plain conversational English, no formatting. Keep it short (1–2 sentences ideal).",
      }),
      voice: Type.Optional(
        Type.String({
          description:
            "Optional voice override. Defaults to am_michael. Other voices: am_fenrir, am_puck, bm_george, af_heart, af_bella, etc.",
        })
      ),
    }),

    async execute(_id, params, _signal, onUpdate, _ctx) {
      const { text, voice } = params as { text: string; voice?: string };
      const r = await speak(text, voice || DEFAULT_VOICE, onUpdate as AgentToolUpdateCallback);
      if (r.ok) {
        return {
          content: [{ type: "text", text: `🔊 Spoke aloud (${r.voice}): "${text}"` }],
          details: r,
        };
      }
      return {
        content: [{ type: "text", text: `❌ TTS playback failed: ${r.error}` }],
        details: r,
        isError: true,
      };
    },
  });

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
      ctx.ui.notify("🎤 Synthesizing speech...", "info");
      const r = await speak(text, voice || DEFAULT_VOICE);
      if (r.ok) {
        ctx.ui.notify(`🔊 Spoke (${r.voice}): "${text}"`, "info");
      } else {
        ctx.ui.notify(`❌ TTS failed: ${r.error}`, "error");
      }
    },
  });
}
