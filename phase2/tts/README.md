# Phase 2 local TTS — AI4Bharat Indic Parler-TTS

Indic Parler-TTS is used because it is designed for Indic languages and English, including Hindi.

## Important model-access requirement

The Hugging Face repository for `ai4bharat/indic-parler-tts` currently requires you to sign in and accept its access conditions. After accepting them, create a Hugging Face access token and authenticate the machine before the first model download.

Typical setup:

```bash
pip install huggingface_hub
hf auth login
```

or set your approved token through the normal Hugging Face environment configuration used by your infrastructure.

## Install

Use a separate environment because TTS has heavier PyTorch/audio dependencies:

```bash
python3 -m venv .venv-tts
source .venv-tts/bin/activate
pip install -r phase2/tts/requirements-server.txt
```

## Run

```bash
INDIC_PARLER_MODEL=ai4bharat/indic-parler-tts \
uvicorn server:app --app-dir phase2/tts --host 0.0.0.0 --port 8003
```

The starter defaults to:

- English speaker: `Thoma`
- Hindi speaker: `Divya`

Configure the main app:

```env
TTS_PROVIDER=indic_parler
LOCAL_TTS_URL=http://localhost:8003/synthesize
LOCAL_TTS_VOICE_EN=Thoma
LOCAL_TTS_VOICE_HI=Divya
```

## TTS adaptation dataset

Only train/adapt a voice when you have clear rights/consent to the recordings.

```json
{"audio":"tts/en_0001.wav","text":"Your refund has been initiated.","speaker_id":"owned_spk_en","language":"en"}
{"audio":"tts/hi_0001.wav","text":"आपका रिफंड शुरू कर दिया गया है।","speaker_id":"owned_spk_hi","language":"hi"}
```

Evaluate intelligibility, pronunciation of domain terms/numbers, naturalness, latency, and speaker consistency separately for English, Hindi and Hinglish-heavy sentences.
