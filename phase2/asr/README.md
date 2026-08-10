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
