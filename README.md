# Voice RAG Assistant — English + Hindi

A two-phase reference project for an English/Hindi document-grounded voice assistant.

**Voice or text → STT → multilingual RAG → Qwen answer → TTS → feedback**

- Backend: **FastAPI**
- Frontend: **React + Vite**
- Phase 1: **Groq Whisper + Groq Qwen + local BGE-M3/Chroma + Edge/browser TTS**
- Phase 2: **self-hosted Faster-Whisper + Qwen3 QLoRA/vLLM + Indic Parler-TTS**

The application language is English/Hindi. Qwen is used as a multilingual open-weight model family; the application's data and output languages remain English/Hindi.

> No speech/LLM pipeline can guarantee 100% accuracy. Measure ASR error rate, retrieval quality, grounded-answer correctness, refusal quality and end-to-end latency on your own data.

## 1. Phase-1 architecture

```text
React microphone/text
       │
       ├── voice -> POST /api/v1/voice-query
       │              │
       │              └── Groq Whisper (en or hi) -> transcript
       │
       └── text  -> POST /api/v1/query
                      │
                      ▼
              BGE-M3 query embedding
                      │
                      ▼
                Chroma vector DB
           English + Hindi documents
                      │
                 top-k chunks
                      │
                      ▼
             Groq Qwen 3.6 27B
        grounded English/Hindi answer
                      │
                      ▼
        Edge TTS en-IN / hi-IN voice
                      │
                      ▼
        React answer + sources + latency
                      │
                      ▼
          feedback + optional correction
```

There is **no translation stage**. A Hindi question can retrieve an English document and still receive a Hindi answer, and vice versa.

## 2. Phase-1 API keys

Default Phase 1 needs only **one Groq API key**. The same key is used for:

- STT: `whisper-large-v3-turbo`
- LLM: `qwen/qwen3.6-27b`

BGE-M3 and Chroma run locally. `edge-tts` does not require an API key in this prototype.

Read: [`docs/API_KEYS.md`](docs/API_KEYS.md).

## 3. Prerequisites

- Python 3.11 recommended
- Node.js 20+
- Git
- internet access for Groq and first-time model/package downloads
- Groq API key

## 4. Install

### Windows PowerShell

```powershell
Expand-Archive .\voice-rag-en-hi.zip -DestinationPath .
cd .\voice-rag-en-hi
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_phase1.ps1
```

### macOS/Linux

```bash
unzip voice-rag-en-hi.zip
cd voice-rag-en-hi
./scripts/setup_phase1.sh
```

The script creates `.env` if missing, creates `.venv`, installs backend dependencies and runs `npm install` in the frontend.

## 5. Configure `.env`

Open the root `.env` and set:

```env
GROQ_API_KEY=gsk_your_key_here
```

For the default Phase 1, leave these values unchanged:

```env
DEFAULT_LANGUAGE=en
SUPPORTED_LANGUAGES=en,hi
STT_PROVIDER=groq
LLM_PROVIDER=groq
TTS_PROVIDER=edge

GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_LLM_MODEL=qwen/qwen3.6-27b

EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_COLLECTION=knowledge
RAG_TOP_K=5
RAG_MIN_SCORE=0.20
CHUNK_SIZE=700
CHUNK_OVERLAP=120

EDGE_TTS_VOICE_EN=en-IN-NeerjaNeural
EDGE_TTS_VOICE_HI=hi-IN-SwaraNeural
```

## 6. Add English/Hindi documents

Put knowledge files here:

```text
backend/data/docs/
```

Supported:

- PDF — text-based PDF
- DOCX — paragraph text
- TXT — UTF-8 recommended
- Markdown
- CSV — UTF-8/UTF-8-BOM recommended

Demo files are already included:

```text
sample_leave_policy_en.md
sample_refund_policy_hi.txt
sample_product_faq_bilingual.csv
```

They are fictional test data; replace/delete them for real business use.

Detailed rules: [`docs/DOCUMENT_GUIDE.md`](docs/DOCUMENT_GUIDE.md).

## 7. Validate and ingest documents

Activate the environment, then inspect extraction quality:

```bash
python backend/scripts/validate_docs.py --path backend/data/docs
```

Rebuild the local Chroma collection:

```bash
python backend/scripts/ingest_docs.py --path backend/data/docs --reset
```

`--reset` recreates the collection. Omit it if you are intentionally adding data without a full rebuild.

## 8. Run backend

macOS/Linux:

```bash
source .venv/bin/activate
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Useful URLs:

```text
Health:   http://localhost:8000/api/v1/health
Swagger:  http://localhost:8000/docs
```

## 9. Test text RAG before voice

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

The response includes the generated answer, retrieved source chunks and retrieval/LLM timing.

## 10. Run frontend

```bash
cd frontend
cp .env.example .env.local
npm run dev
```

PowerShell:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm run dev
```

Open:

```text
http://localhost:5173
```

Choose **English** or **Hindi** and use either the microphone or the text box.

## 11. Voice endpoint

```bash
curl -X POST http://localhost:8000/api/v1/voice-query \
  -F 'audio=@sample.webm' \
  -F 'language=en' \
  -F 'top_k=5'
```

For Hindi use `language=hi`.

The voice response includes:

- transcript
- grounded answer
- retrieved sources
- STT/retrieval/LLM/TTS/total timings
- base64 audio when Edge TTS succeeds
- browser-TTS fallback flag

## 12. Document upload endpoint

While the API is running:

```bash
curl -F "file=@my_policy.pdf" http://localhost:8000/api/v1/documents
```

Add authentication/authorization before enabling this endpoint for real users.

## 13. Feedback

The React UI supports thumbs up/down and optional corrected answers. Feedback is stored in:

```text
backend/data/feedback/feedback.jsonl
```

Phase-2 scripts can convert reviewed feedback into SFT/DPO data.

## 14. Phase 2

Phase 2 keeps the same React and main FastAPI API contracts but changes model providers:

```text
ASR: Faster-Whisper self-hosted
SLM: fine-tuned Qwen3 served through vLLM
RAG: BGE-M3 + production vector store when needed
TTS: AI4Bharat Indic Parler-TTS
```

Example provider switch:

```env
STT_PROVIDER=faster_whisper
LLM_PROVIDER=openai_compatible
TTS_PROVIDER=indic_parler

LOCAL_ASR_URL=http://localhost:8002/transcribe
LOCAL_LLM_BASE_URL=http://localhost:8001/v1
LOCAL_LLM_MODEL=qwen-domain
LOCAL_TTS_URL=http://localhost:8003/synthesize
```

Read [`phase2/README.md`](phase2/README.md).

## 15. RAG vs fine-tuning

Do not fine-tune the model to memorize changing PDFs.

Use **RAG** for:

- policies
- product facts
- manuals
- SOPs
- frequently changing knowledge

Use **fine-tuning** for:

- English/Hindi domain terminology
- Hinglish style/technical-term handling
- preferred answer structure
- recurring instruction-following failures
- verified corrected responses

## 16. Repository map

```text
backend/
  app/                  FastAPI, providers, RAG, orchestration
  data/docs/            English/Hindi RAG documents + samples
  data/chroma/          local vector DB
  data/feedback/        feedback JSONL
  data/eval/            sample RAG QA labels
  scripts/              ingestion and document validation
frontend/               React microphone + text UI
scripts/                setup + Groq key verification
phase2/                 QLoRA, local ASR/TTS, evaluation
 docs/                   setup, architecture, data, costs, deployment
```

## 17. Security/production notes

- Never put API keys in React.
- Do not commit `.env`.
- Add authentication/authorization before document upload or private knowledge access.
- Production RAG should enforce document-level access controls/metadata filters.
- Define retention and encryption rules for sensitive data and feedback.
- The starter does not persist raw microphone audio.
- `edge-tts` is convenient for prototyping but is not a production SLA-backed API.
- Scanned/image-only PDFs require OCR before the current loader can use them.
