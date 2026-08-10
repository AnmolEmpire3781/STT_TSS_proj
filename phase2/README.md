# Phase 2 — self-hosted English/Hindi models, dataset and fine-tuning

Phase 2 changes providers, not the public application API.

## Recommended reference stack

| Layer | Reference choice | Purpose |
|---|---|---|
| ASR | Faster-Whisper `large-v3-turbo` | runnable multilingual English/Hindi local ASR |
| Hindi ASR benchmark | AI4Bharat IndicConformer | compare on your Hindi/accent/domain test set before production selection |
| SLM | Qwen3-1.7B then Qwen3-4B | multilingual open-weight model suitable for QLoRA experiments |
| Embedding | BGE-M3 | English/Hindi and cross-lingual RAG |
| Vector DB | Chroma → Qdrant/pgvector | MVP → production persistence/scaling |
| LLM serving | vLLM OpenAI-compatible server | stable provider contract and batching |
| TTS | AI4Bharat Indic Parler-TTS | English + Hindi/Indic TTS |

## A. Keep datasets separate by purpose

```text
documents/               -> RAG knowledge corpus
asr_manifest.jsonl       -> audio path + exact human transcript
sft_train.jsonl          -> instruction/query + ideal assistant answer
sft_eval.jsonl           -> frozen held-out behavioral set
dpo_train.jsonl          -> prompt + chosen + rejected (optional)
tts_manifest.jsonl       -> authorized audio + transcript + speaker metadata
```

Do not train the Qwen adapter on every PDF to memorize facts. Changing facts belong in RAG.

## B. Feedback -> SFT/DPO

```bash
python phase2/datasets/build_sft_from_feedback.py \
  --input backend/data/feedback/feedback.jsonl \
  --output phase2/datasets/generated/sft_from_feedback.jsonl

python phase2/datasets/build_dpo_from_feedback.py \
  --input backend/data/feedback/feedback.jsonl \
  --output phase2/datasets/generated/dpo_from_feedback.jsonl
```

Use reviewed examples only. Negative feedback without a verified correction should not automatically become a training target.

## C. Fine-tune Qwen3 with QLoRA

Install on an NVIDIA/CUDA training machine:

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r phase2/requirements-train.txt
```

Smoke test:

```bash
python phase2/train/train_qwen_qlora.py \
  --model Qwen/Qwen3-0.6B \
  --train phase2/datasets/examples/sft_en_hi.jsonl \
  --output phase2/output/smoke-lora \
  --epochs 1 \
  --max-steps 5
```

First real experiment:

```bash
python phase2/train/train_qwen_qlora.py \
  --model Qwen/Qwen3-1.7B \
  --train phase2/datasets/generated/sft_from_feedback.jsonl \
  --output phase2/output/qwen3-1.7b-domain-lora \
  --epochs 2
```

Approximate planning targets, not guarantees:

- 1.7B QLoRA: roughly 12–16 GB VRAM is a comfortable initial target around ~2k context
- 4B QLoRA: roughly 16–24 GB VRAM target

Actual memory depends on context length, batch/accumulation, library versions and GPU.

## D. Merge and serve Qwen

```bash
python phase2/train/merge_lora.py \
  --base Qwen/Qwen3-1.7B \
  --adapter phase2/output/qwen3-1.7b-domain-lora \
  --output phase2/output/qwen3-1.7b-domain-merged
```

```bash
pip install vllm
vllm serve phase2/output/qwen3-1.7b-domain-merged \
  --served-model-name qwen-domain \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096
```

Main app:

```env
LLM_PROVIDER=openai_compatible
LOCAL_LLM_BASE_URL=http://localhost:8001/v1
LOCAL_LLM_MODEL=qwen-domain
```

## E. Run local Faster-Whisper ASR

See [`asr/README.md`](asr/README.md).

```bash
uvicorn server:app --app-dir phase2/asr --host 0.0.0.0 --port 8002
```

Then:

```env
STT_PROVIDER=faster_whisper
LOCAL_ASR_URL=http://localhost:8002/transcribe
```

Benchmark it separately on English, Hindi and Hinglish. If Hindi domain recognition is not sufficient, benchmark AI4Bharat IndicConformer rather than assuming one model is best.

## F. Run Indic Parler-TTS

See [`tts/README.md`](tts/README.md).

The current Hugging Face Indic Parler model repository is gated: accept its conditions and authenticate the Phase-2 machine before downloading it.

Then:

```env
TTS_PROVIDER=indic_parler
LOCAL_TTS_URL=http://localhost:8003/synthesize
LOCAL_TTS_VOICE_EN=Thoma
LOCAL_TTS_VOICE_HI=Divya
```

## G. Evaluation

Track separate slices:

```text
English
Hindi
Hinglish/code switching
cross-language RAG (Hindi query -> English doc)
cross-language RAG (English query -> Hindi doc)
no-answer/refusal questions
noise/accent/device ASR slices
```

Run:

```bash
python phase2/eval/wer.py --input phase2/eval/example_asr_eval.jsonl
python phase2/eval/cer.py --input phase2/eval/example_asr_eval.jsonl
```

Also measure retrieval Hit@1/3/5, grounded correctness, refusal precision and P50/P95 latency by stage.

## H. Data hosting/connectors

Start locally for development. For team/production data use an approved store:

- private S3/GCS/Azure Blob/MinIO for documents/audio
- Hugging Face Hub for versioned datasets/models only when policy permits
- Git/Git LFS only for tiny non-sensitive fixtures
- DVC/lakeFS when dataset lineage is required

Log each training run's dataset version/hash, base-model revision, adapter config, code commit, evaluation result and artifact checksum.

## I. Optimization order

1. parsing/data quality
2. retrieval quality
3. grounding/refusal prompt
4. ASR errors for domain terms/numbers
5. SLM fine-tuning
6. TTS pronunciation
7. quantization/batching/streaming
8. WebSocket/WebRTC + VAD + streamed LLM/TTS when latency requires it
