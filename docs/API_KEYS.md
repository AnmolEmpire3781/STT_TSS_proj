# Phase 1 API keys

## Required: Groq API key

The default Phase-1 stack uses one Groq API key for both:

- Speech-to-text: `whisper-large-v3-turbo`
- LLM: `qwen/qwen3.6-27b`

Create/manage it in the Groq Console under **API Keys** for the selected Groq project. Do not put the key in React or commit it to Git.

Set it in the root `.env`:

```env
GROQ_API_KEY=gsk_your_real_key_here
```

### Verify after editing `.env`

macOS/Linux:

```bash
python scripts/check_groq_key.py
```

Windows PowerShell:

```powershell
py scripts\check_groq_key.py
```

The script reads `GROQ_API_KEY` from the terminal first and otherwise from the root `.env`. It calls Groq's `/models` endpoint and checks whether the configured STT/LLM model IDs appear.

## Not required in Phase 1

No API key is required for:

- BGE-M3 embeddings — downloaded and run locally through `sentence-transformers`
- Chroma — local persistent vector database
- `edge-tts` — no API key in this prototype adapter
- React/FastAPI

If your network requires authentication to download public Hugging Face models, configure that separately according to your environment; it is not part of the default project contract.
