# pi-voice-michael

Give your Pi agent a voice. Offline TTS via **Kokoro ONNX** (82M params, neural quality) with the **am_michael** US English male voice — fully local, no API keys, no cloud.

## What you get

- **`voice_say_aloud`** tool — your agent can speak text aloud through your speakers
- **`/say` command** — quickly test TTS from the Pi TUI
- **`agent_end` auto-speak** — agent says a final summary after every response (opt-in)

## Why Kokoro am_michael?

Kokoro is a frontier TTS model for its size (82M params), freely licensed (Apache), and runs entirely locally via ONNX Runtime. The **am_michael** voice is a clear, professional US English male — comparable quality to models 5–10x its size.

| Quality | tiny-tts | **Kokoro (am_michael)** |
|---------|----------|-------------------------|
| Sample rate | 44.1 kHz Float | 24 kHz Float |
| Params | 1.6M | 82M |
| Naturalness | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| RAM (typical) | ~80 MB | ~200–400 MB |
| Disk cache | ~6 MB | ~90 MB |

## Requirements

- Node.js ≥ 22
- One of these audio players on your `$PATH`:
  - **Linux**: `paplay` (PulseAudio), `aplay` (ALSA), or `pw-play` (PipeWire)
  - **macOS**: `afplay` (built-in)
  - **Windows**: PowerShell + `ffplay`

## Installation

```bash
# From GitHub (after release)
pi install github:steimerbyte/pi-voice-michael

# Or local path for development
pi install /home/steimerbyte/dev/pi-voice-michael
```

## Usage

Once installed, your agent sees this tool:

```
voice_say_aloud(text: string) → spoken audio plays through speakers
```

Example prompt to your agent:

> "Sprich mir mit dem Michael-Tool einen kurzen Bauleiter-Status vor."

The agent will call:
```ts
voice_say_aloud({ text: "Alles erledigt. Build grün, Tests grün, bereit für Deployment." })
```

And the audio plays on your machine.

### Test the TTS directly

```
/say Build complete, all tests passing.
```

## License

MIT
