# Voice RAG Repository Instructions

## Core architecture

This project is an English/Hindi voice RAG application.

Stable voice workflow:

Record
-> Stop
-> Ask with voice
-> STT
-> RAG or Direct LLM
-> TTS
-> response

Do not replace this with WebSocket, VAD, continuous listening,
always-on microphone, or automatic silence submission unless the
user explicitly requests it.

## Supported answer modes

The application has two explicit modes.

### Document RAG mode

rag_based=true

Flow:

query
-> BGE-M3 embedding
-> Chroma retrieval
-> top-k document chunks
-> LLM
-> answer

Document-specific and organization-specific answers must be grounded
in retrieved documents.

### Direct LLM mode

rag_based=false

Flow:

query
-> LLM directly
-> answer

Direct mode must skip BGE-M3 and Chroma completely.

In direct mode:

- sources must be empty
- retrieval timing should be zero
- do not claim access to private documents
- do not invent organization-specific facts

rag_based=true must remain the default for backward compatibility.

## Current providers

STT:
Groq whisper-large-v3-turbo

LLM:
Groq OpenAI-compatible endpoint using configured Qwen model

Embedding:
BAAI/bge-m3

Vector database:
Chroma

TTS:
Edge TTS

Do not replace providers, models, ports, API URLs, document paths,
or environment settings unless explicitly requested.

## Languages

Supported response languages:

- English
- Hindi

Hinglish and mixed Hindi-English queries should be understood
naturally.

Do not translate technical names unnecessarily.

## Output requirements

Responses are used by TTS.

LLM output must pass through:

app.services.text_cleanup.clean_assistant_answer

Avoid:

- Markdown symbols
- [S1] style source markers in spoken answers
- LaTeX delimiters
- URLs unless explicitly requested
- formatting symbols that TTS may read aloud

Sources remain available separately in the response metadata/UI.

## Human feedback

Feedback must retain:

- request_id
- language
- rating
- query
- model_answer
- corrected_answer
- note
- sources
- rag_based

Unreviewed user feedback must never automatically modify model
weights or override official documents.

## Change discipline

Before editing:

1. Inspect the relevant existing files.
2. Understand current call paths.
3. Preserve existing working behavior not involved in the request.

When implementing:

- make minimal focused changes
- do not rewrite unrelated files
- keep useful comments
- add comments around non-obvious architectural decisions
- preserve existing naming and style where practical

For requests to explain/review:

- inspect and report
- do not modify files unless explicitly asked

For requests to implement/fix:

- make the requested local changes
- run relevant non-destructive validation automatically

Ask before:

- deleting important data
- destructive Git operations
- force pushes
- changing secrets
- external deployment
- large dependency upgrades
- material scope expansion

## Required validation

For backend changes run appropriate checks such as:

python -m py_compile <changed python files>

and:

python -c "import sys; sys.path.insert(0,'backend'); from app.main import app; print('BACKEND IMPORT OK')"

For frontend changes run:

npm run build

When practical also run targeted functional tests.

At completion report:

- files changed
- behavior changed
- validation performed
- anything not tested
