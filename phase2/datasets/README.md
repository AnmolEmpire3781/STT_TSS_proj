# Dataset lifecycle

Keep four concerns separate:

1. **Knowledge corpus** — PDFs/DOCX/CSV used by RAG; version these independently from model training data.
2. **SFT data** — curated instruction/response examples for behavior and domain terminology.
3. **Preference data** — chosen/rejected answer pairs from human review, used only after SFT is stable.
4. **Speech data** — ASR/TTS audio manifests with exact human transcripts and legal consent/usage rights.

## Suggested versioning layout

```text
datasets/
  releases/
    2026-08-01/
      sft_train.jsonl
      sft_eval.jsonl
      dpo_train.jsonl
      asr_train.jsonl
      asr_eval.jsonl
      manifest.sha256
```

Never overwrite an already-used training release. Create a new dated/versioned release and log its hash with the model artifact.

## Build from production feedback

```bash
python phase2/datasets/build_sft_from_feedback.py \
  --input backend/data/feedback/feedback.jsonl \
  --output phase2/datasets/generated/sft.jsonl

python phase2/datasets/build_dpo_from_feedback.py \
  --input backend/data/feedback/feedback.jsonl \
  --output phase2/datasets/generated/dpo.jsonl
```

Human review is still required before training.

## Optional Hugging Face Hub connector

For datasets your organization is allowed to host on Hugging Face:

```bash
export HF_TOKEN=hf_...
python phase2/datasets/publish_hf_dataset.py \
  --train phase2/datasets/generated/sft.jsonl \
  --repo your-org/voice-rag-domain-sft \
  --private
```

If your data is sensitive, prefer your approved private object store (S3/GCS/Azure Blob/MinIO) and keep only hashes/metadata in the ML experiment tracker.
