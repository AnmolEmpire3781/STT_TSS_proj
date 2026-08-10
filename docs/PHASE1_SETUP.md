# Phase 1 complete setup — English + Hindi

## 1. Prerequisites

- Python 3.11 recommended
- Node.js 20+
- Git
- Internet on first setup/download
- one Groq API key

## 2. Configure environment

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace only:

```env
GROQ_API_KEY=replace_me
```

For the default Phase-1 path, leave these as-is:

```env
DEFAULT_LANGUAGE=en
SUPPORTED_LANGUAGES=en,hi
STT_PROVIDER=groq
LLM_PROVIDER=groq
TTS_PROVIDER=edge
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_LLM_MODEL=qwen/qwen3.6-27b
EMBEDDING_MODEL=BAAI/bge-m3
```

## 3. Install

macOS/Linux:

```bash
./scripts/setup_phase1.sh
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_phase1.ps1
```

## 4. Verify Groq key

macOS/Linux:

```bash
source .venv/bin/activate
python scripts/check_groq_key.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\check_groq_key.py
```

The verification script reads the root `.env`, so no extra terminal export is required.

## 5. Add documents

Place `.pdf`, `.docx`, `.txt`, `.md`, or `.csv` files under:

```text
backend/data/docs/
```

Sample English/Hindi/bilingual demo files are included. Replace them with your own data for real testing.

Validate extraction:

```bash
python backend/scripts/validate_docs.py --path backend/data/docs
```

## 6. Build/rebuild Chroma knowledge index

```bash
python backend/scripts/ingest_docs.py --path backend/data/docs --reset
```

`--reset` deletes/recreates the collection before ingesting everything. Omit it when adding a file without rebuilding the whole collection.

## 7. Start backend

macOS/Linux:

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Health:

```text
http://localhost:8000/api/v1/health
```

Interactive API docs:

```text
http://localhost:8000/docs
```

## 8. Test text RAG before microphone

English:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many paid annual leave days do employees receive?","language":"en","top_k":5}'
```

Hindi:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"रिफंड मिलने में कितना समय लग सकता है?","language":"hi","top_k":5}'
```

## 9. Start frontend

```bash
cd frontend
cp .env.example .env.local
npm run dev
```

Windows PowerShell uses the same `npm run dev` command after `Copy-Item .env.example .env.local`.

Open:

```text
http://localhost:5173
```

Choose **English** or **Hindi**, then use voice or text.

## 10. Voice request flow

```text
React MediaRecorder
  -> POST /api/v1/voice-query
  -> Groq Whisper transcription with language=en|hi
  -> BGE-M3 query embedding
  -> Chroma top-k semantic retrieval over English/Hindi docs
  -> Qwen question + retrieved chunks + requested output language
  -> edge-tts English/Hindi voice
  -> React plays audio and displays sources/timings
  -> optional thumbs up/down + correction stored in feedback JSONL
```

## 11. TTS voice troubleshooting

List voices from the installed package:

```bash
edge-tts --list-voices
```

If a configured voice is unavailable, select an `en-IN` and `hi-IN` voice from that list and update:

```env
EDGE_TTS_VOICE_EN=...
EDGE_TTS_VOICE_HI=...
```

If Edge TTS fails at request time, the frontend falls back to browser `speechSynthesis`.
