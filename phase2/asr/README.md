# Phase 2 local ASR — Faster-Whisper

The runnable self-hosted reference service uses **Faster-Whisper** because it supports English/Hindi through multilingual Whisper and has a simple production-friendly CTranslate2 inference path.

AI4Bharat IndicConformer remains a useful Hindi-specific benchmark candidate, but it is not wired into the starter because model/release-specific interfaces should be benchmarked before replacing the stable HTTP contract.

## Install

```bash
python3 -m venv .venv-asr
source .venv-asr/bin/activate
pip install -r phase2/asr/requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv-asr
.\.venv-asr\Scripts\Activate.ps1
pip install -r phase2\asr\requirements.txt
```

## Run

CPU-friendly first test:

```bash
FASTER_WHISPER_MODEL=small \
FASTER_WHISPER_DEVICE=cpu \
FASTER_WHISPER_COMPUTE_TYPE=int8 \
uvicorn server:app --app-dir phase2/asr --host 0.0.0.0 --port 8002
```

GPU target:

```bash
FASTER_WHISPER_MODEL=large-v3-turbo \
FASTER_WHISPER_DEVICE=cuda \
FASTER_WHISPER_COMPUTE_TYPE=float16 \
uvicorn server:app --app-dir phase2/asr --host 0.0.0.0 --port 8002
```

Configure the main application:

```env
STT_PROVIDER=faster_whisper
LOCAL_ASR_URL=http://localhost:8002/transcribe
```

## ASR training/adaptation data

A future ASR training manifest should contain legally collected audio and exact human transcripts:

```json
{"audio":"audio/en_0001.wav","text":"Please reset my VPN connection.","language":"en","speaker_id":"spk001"}
{"audio":"audio/hi_0001.wav","text":"मेरा VPN connection reset कर दीजिए।","language":"hi","speaker_id":"spk002"}
```

Keep train/dev/test speakers separated. Include Indian English accents, Hindi accents, office/noise conditions, domain names, acronyms, numbers and Hinglish/code-switching.

Evaluate both WER and task-specific entity accuracy; for Devanagari-heavy Hindi you can additionally track normalized CER.

## Offline Phase-2 training and evaluation

The offline pipeline is separate from `server.py` and does not change the current
production STT provider. Its default persistent root is
`/content/drive/MyDrive/voice-rag-phase2/stt`; every command that writes there
accepts `--drive-root` or an explicit output path. Disposable staging defaults
to `/content/voice-rag-asr`.

Install the normal training environment with:

```bash
pip install -r phase2/asr/requirements-train.txt
```

Install `requirements-8bit.txt` only when using `--load-in-8bit`.

Public training data must be created with `sample_public_subset.py`. It accepts
only the protected IndicVoices Hindi and MUCS train splits, streams a pinned Hub
revision, and persists only the selected hour snapshot. `freeze_eval_suite.py`
creates the immutable benchmark500 suite from IndicVoices valid, MUCS test, and
Svarah test. Custom releases created by `prepare_custom_dataset.py` are
speaker-disjoint and require `human_verified=true`; the trainer filters their
single sealed manifest to train and validation rows without admitting test rows.

For Hinglish, evaluate the base model three times with `--hinglish-mode auto`,
`hi`, and `en`. Record the empirical winner from the fixed benchmark with:

```bash
python phase2/asr/compare_models.py \
  --select-hinglish-from "$AUTO_EVAL" "$HI_EVAL" "$EN_EVAL" \
  --output-dir "$DRIVE_ROOT/evaluation/decoding/benchmark500-v1"
```

Use the resulting immutable selection for tuned evaluation:

```bash
python phase2/asr/evaluate_whisper.py \
  --suite "$DRIVE_ROOT/evaluation/suites/benchmark500-v1" \
  --profile benchmark500 \
  --adapter-path "$DRIVE_ROOT/exports/RUN_ID/adapter" \
  --decoding-config "$DRIVE_ROOT/evaluation/decoding/benchmark500-v1/selected-decoding.json" \
  --staging-dir "/content/voice-rag-asr/eval-tuned" \
  --output-dir "$DRIVE_ROOT/evaluation/runs/RUN_ID"
```

`compare_models.py --base ... --candidate ...` rejects different suite,
decoding, normalization, keyword, prediction-order, or latency configurations.
Training and evaluation manifests are checksum-verified by default; the
incomplete-manifest escape hatch is explicitly development-only. Checkpoint
retention defaults to two compatible checkpoints, and cleanup remains a dry run
unless `cleanup_colab.py --execute` is supplied.
