# Evaluation plan — English + Hindi

Create a frozen evaluation set before fine-tuning.

## ASR

Track WER by slice and optionally normalized CER for Hindi/Devanagari-heavy tests.

Required slices:

- English / Indian English accents
- Hindi
- Hinglish/code switching
- clean microphone
- office/background noise
- different devices
- product/domain names and acronyms
- numbers, dates, amounts and IDs

Scripts:

```bash
python phase2/eval/wer.py --input phase2/eval/example_asr_eval.jsonl
python phase2/eval/cer.py --input phase2/eval/example_asr_eval.jsonl
```

## Retrieval

Build at least 100 labeled questions with expected source files/chunks. Include:

- English query -> English doc
- Hindi query -> Hindi doc
- Hindi query -> English doc
- English query -> Hindi doc
- Hinglish query
- explicit no-answer cases

Track Hit@1, Hit@3, Hit@5, Recall@K and false-positive retrieval on no-answer questions.

## LLM

Store an acceptable answer, required source IDs, refusal expectation and critical facts. Measure grounded correctness separately from style/fluency.

## TTS

Use listening tests for English, Hindi and Hinglish-heavy text:

- intelligibility
- naturalness
- domain-term pronunciation
- numbers/acronyms
- prosody
- speaker consistency

## End-to-end

Record P50/P95 for STT, retrieval, LLM, TTS and total latency. Optimize the measured bottleneck rather than guessing.
