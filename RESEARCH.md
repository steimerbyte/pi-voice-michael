# TTS Engine Research — 2026-08-12

## Frage: Gibt es Engines mit gleicher Qualität wie Kokoro die lange Texte ohne 30s-Chunking handlen?

## Kokoro Limit (was wir jetzt nutzen)

**Hard limit:** 510 phoneme tokens pro generate() (~500 chars input, ~30s audio output)

Quellen:
- kokorottsai.com: "Can Kokoro TTS handle long text inputs? ... up to 510 tokens in a single pass"
- nolist.ai: "Audio quality takes a sharp dive if you exceed the 510-token context"
- remsky/Kokoro-FastAPI: caps chunks at 450 tokens, "running it that long tends to produce 'rushed' speech and other artifacts"
- NVIDIA model card: "recommends splitting into 100-200-token chunks"
- qwe.edu.pl: "Max input is around 500-510 tokens per pass"

→ Best practice: chunks von 100-200 tokens, NICHT 510. Bei 10 Sätze pro chunk sind wir bei ~100-250 tokens pro chunk — gut.

→ Mein aktuelles Staged-Pipeline mit CONCURRENCY=2 + mpv gapless ist state-of-the-art.

## Alternative Engines

### Piper (Python, neural VITS)

**Pros:**
- Echtes streaming: `piper --output-raw | aplay` mit sentence-by-sentence synthesis
- "POST /api/v1/stream synthesizes sentence-by-sentence and returns chunked WAV as each segment completes"
- TTFA (time-to-first-audio): under 1 second in production
- 22kHz, gute Qualität, deutsche Stimmen nativ
- 50+ Stimmen, multi-language

**Cons:**
- Python dependency (pip install piper-tts)
- Eigene piper binary nötig
- Nicht in node.js ohne wrapper (kokoro-js-style)

**Verdict:** Wenn man Python als dependency akzeptiert, **besser als Kokoro** für sehr lange Texte. Aber: zusätzliches Setup, npm-paket overhead.

### Coqui XTTS v2

**Pros:**
- **Kein hartes Token-Limit** — kann beliebig lange Texte in einem Aufruf
- Voice cloning (6-20 sec sample)
- 17 Sprachen
- 100% lokal

**Cons:**
- **Python-only**
- 1.8GB Model
- ~1GB RAM minimum
- Langsamer als Kokoro (RTF 0.5-1.0)

**Verdict:** Beste für sehr lange Texte ohne chunking, aber overhead hoch.

### Silero TTS

**Pros:**
- Klein, schnell, CPU-only
- Open-source Modelle verfügbar

**Cons:**
- Weniger ausgereift als Piper/Kokoro
- Wenig Node.js-Bindings

**Verdict:** Solide aber nicht besser als Kokoro.

### FlashTTS (arxiv 2606.09141)

**Pros:**
- **325ms First-Packet Latency** (vs Kokoro's ~30s für ersten chunk)
- Multi-Token Prediction (MTP) Acceleration
- X-pred Mean Flow Distillation

**Cons:**
- **Akademisches Paper, kein öffentliches npm/python package**
- Qwen2-0.5B backbone = größer als Kokoro

**Verdict:** Zukunftsmusik, noch nicht produktionsreif.

### SpeakStream (arxiv 2505.19206)

**Pros:**
- Echtes sentence-level streaming
- Decoder-only dual-streaming

**Cons:**
- Paper, kein public release

### Cartesia Sonic

**Pros:**
- **40-100ms TTFB** (schnellste am Markt)
- SSM-Architektur, native streaming

**Cons:**
- **Cloud API**, kein offline
- Kommerziell

### ElevenLabs Flash

**Pros:**
- 75ms TTFA
- Beste Audio-Qualität am Markt

**Cons:**
- **Cloud**, $5-$330/mo
- Nicht offline

## Was die Recherche ergibt

**Für unseren Use-Case (offline, Node.js, deutsche Texte, pi-agent):**

| Kriterium | Kokoro (current) | Piper | XTTS v2 |
|-----------|------------------|-------|---------|
| Offline | ✅ | ✅ (Python) | ✅ (Python) |
| Pure Node.js | ✅ | ❌ | ❌ |
| Multi-language | ✅ 9 langs | ✅ 50+ | ✅ 17 |
| Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Streaming | ⚠️ 30s chunks | ✅ sentence | ✅ unlimited |
| Hard token limit | 510 | none | none |
| RAM | ~200MB | ~150MB | ~1GB |
| Setup | npm install | pip+model | pip+1.8GB model |

## Empfehlung

**Bleiben wir bei Kokoro**, aber **tunen die Chunking-Strategie**:

1. **NVIDIA empfohlen: 100-200 tokens pro chunk** (wir sind bei ~100-250 → gut)
2. **Gap zwischen chunks**: 0.3-0.5 sec (menschlicher klingt, vermeidet "rush"-Effekt)
3. **Audio-Konkat** am Ende als optional fallback (für Speicherung)
4. **Voice-Switch Limit**: Kokoro kann pro chunk eine andere voice nutzen — Multi-Voice passages möglich

**Alternative (wenn Python akzeptabel):** Piper via Python subprocess im Hintergrund:
```javascript
const piper = spawn('piper', ['--model', 'de_DE-thorsten-low.onnx', '--output-raw']);
piper.stdin.write(text);
piper.stdout.on('data', chunk => paplay.write(chunk));  // streaming!
```

Das wäre das **einzige Setup das echtes sentence-level streaming** bietet. Aber: Piper Model-Download (~50MB), Python-Installation, piper binary bauen.

**Für den aktuellen Use-Case**: Kokoro + mein staged-pipeline ist der beste Trade-off zwischen Setup-Komplexität und Streaming-UX.

## Empfehlung: Pipeline-Tuning

Konkret verbessern:
1. Chunk size: von "10 Sätze" auf "**bis zu 200 tokens**" umstellen (NVIDIA-empfohlen)
2. Gap zwischen chunks: 300ms silent hinzufügen
3. Erste chunk priorisieren: < 100 tokens für schnellsten ersten Audio
4. Letzter chunk: stretch auf 200 tokens wenn übrig
5. Streaming-CLI: `--no-cache-on-disk` vermeidet disk-thrashing

Konkreter Plan: Update `speak()` mit Token-basierter Chunker statt Satz-basierter.