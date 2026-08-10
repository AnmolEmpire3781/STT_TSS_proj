# Cost model (checked August 2026)

## Phase 1 default

| Component | Default | Prototype cost |
|---|---|---|
| STT | Groq Free + `whisper-large-v3-turbo` | $0 while you remain inside Free-plan limits |
| LLM | Groq Free + `qwen/qwen3.6-27b` | $0 while inside Free-plan limits |
| Embeddings | `BAAI/bge-m3` locally | $0 API cost; uses your CPU/GPU/RAM |
| Vector DB | Chroma local | $0 API cost |
| TTS | `edge-tts` | no API key; unofficial/no SLA, use only for prototype |
| Frontend/backend | local machine | $0 hosting cost |

### If you upgrade Groq to billed Developer usage

At the time this template was prepared, Groq documented Free-plan limits including 30 RPM / 1,000 RPD for `qwen/qwen3.6-27b`, and 20 RPM / 2,000 RPD for each Whisper model. Free-tier limits can change.

At billed Developer rates:

- `whisper-large-v3-turbo`: **$0.04 per audio hour**
- `whisper-large-v3`: **$0.111 per audio hour**
- `qwen/qwen3.6-27b`: **$0.60 / 1M input tokens** and **$3.00 / 1M output tokens**

Example only: 1,000 voice questions/month, each with 15 seconds of audio, 2,000 input tokens (including RAG context), and 250 output tokens:

- audio: 4.17 hours × $0.04 ≈ **$0.17**
- LLM input: 2M tokens × $0.60/M ≈ **$1.20**
- LLM output: 0.25M × $3.00/M ≈ **$0.75**
- approximate model API spend: **$2.12/month**, before taxes and any other services

Always re-check provider pricing before production because hosted model IDs, limits and prices change.

## Phase 2

There are no per-request model API fees if everything is self-hosted, but you pay for:

- GPU compute (or own the hardware)
- storage for model weights, datasets, logs and vector indexes
- object storage / backups if used
- monitoring/network/egress depending on hosting provider

For a first QLoRA experiment, Qwen3-1.7B is deliberately chosen so that a single consumer/server GPU can be enough. Qwen3-4B gives more headroom in answer quality but needs more VRAM.

## TTS warning

`edge-tts` is a community client for Microsoft Edge's online TTS service. It is convenient for a no-key MVP but it is **not** the production contract/SLA you should base a commercial service on. For the English/Hindi roadmap, replace it with self-hosted Indic Parler-TTS in Phase 2 or an official SLA-backed speech provider.
