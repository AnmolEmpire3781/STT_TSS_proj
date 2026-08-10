# Known limitations

- This is a runnable reference MVP, not a pre-certified production system.
- `edge-tts` is a no-key community client without a production SLA; React can fall back to browser speech synthesis.
- Voice requests are utterance-based, not full-duplex streaming. Add WebSocket/WebRTC/VAD and streaming TTS after measuring the baseline.
- BGE-M3 downloads on first use and can make the first run slow; warm/cache it in deployment.
- Chroma is appropriate for this starter; multi-replica production may need Qdrant, pgvector or another vector service.
- The current PDF parser does not OCR image-only/scanned PDFs.
- DOCX extraction focuses on paragraph text and does not fully parse complex tables/images.
- Large dynamic databases should use database/tool connectors rather than embedding every row.
- Indic Parler-TTS is a heavier Phase-2 model and its current Hugging Face repository requires accepting access conditions before model download.
- ASR/TTS/fine-tuning procedures are version-sensitive. Pin model/repository revisions in production.
- No model provides 100% speech or answer accuracy. Critical workflows require thresholds, refusal/fallback paths and human escalation.
