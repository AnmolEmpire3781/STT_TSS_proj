# Deployment path

## Phase 1 MVP

- React on local/static host
- FastAPI on one CPU VM/container
- local Chroma persistent disk
- Groq hosted STT + LLM
- Edge/browser TTS

## Phase 2 lab

- main FastAPI on CPU
- vLLM Qwen on GPU
- Faster-Whisper ASR on CPU or GPU depending latency target
- Indic Parler-TTS on GPU recommended
- vector DB on CPU/storage service

## Production split

```text
Load balancer
  -> FastAPI replicas
       -> ASR pool
       -> vector DB / retrieval service
       -> vLLM Qwen pool
       -> TTS pool
       -> telemetry + feedback store
```

Operational controls:

- propagated request IDs
- structured JSON logs
- Prometheus/OpenTelemetry
- max upload/audio duration
- timeouts/circuit breakers
- per-user rate limits
- GPU queues/backpressure
- model/version metadata in logs
- authentication and document-level authorization
