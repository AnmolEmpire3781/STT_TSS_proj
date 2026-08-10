# Architecture and latency plan

## Request path

1. User selects English or Hindi.
2. Browser captures microphone audio using `MediaRecorder`, or user types a question.
3. Voice requests go to `/voice-query`; text requests go to `/query`.
4. STT transcribes voice in the selected language.
5. Retriever embeds the question with BGE-M3 and gets the best English/Hindi document chunks from Chroma.
6. LLM receives the requested answer language, strict grounding rules, question and retrieved chunk IDs.
7. Voice requests pass the final answer to the matching English/Hindi TTS voice.
8. API returns transcript, answer, sources and stage timings; voice calls also return audio when available.
9. User feedback is recorded with the request ID and optional corrected answer.

No translation service sits between the query and documents. Multilingual embeddings + multilingual LLM allow cross-language retrieval/generation.

## Provider adapters

```text
STTProvider.transcribe(audio, language) -> text
LLMProvider.answer(question, contexts, language) -> answer
TTSProvider.synthesize(text, language) -> audio bytes
```

Phase 1 uses hosted Groq for STT/LLM. Phase 2 keeps the API and swaps providers for self-hosted model services.

## Latency optimization order

1. keep spoken answers concise
2. avoid heavy reasoning mode for normal grounded FAQ turns
3. tune top-k around 3–5 based on retrieval evaluation
4. precompute document embeddings
5. warm embedding/model services
6. persistent HTTP/model processes
7. add VAD/end-of-speech detection
8. stream LLM output into TTS
9. cache repeated safe queries where business policy permits
10. separate ASR/LLM/TTS GPU worker pools as concurrency grows

## RAG vs fine-tuning

- RAG: changing facts, policies, manuals, product information, approved reference data.
- Fine-tuning: preferred English/Hindi/Hinglish behavior, domain vocabulary, formatting, repeated corrected failures.
