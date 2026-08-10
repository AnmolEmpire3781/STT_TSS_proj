# import base64
# import re
# import time
# from pathlib import Path

# from fastapi import (
#     APIRouter,
#     File,
#     Form,
#     HTTPException,
#     UploadFile,
#     WebSocket,
#     WebSocketDisconnect,
# )

# from app.core.config import get_settings
# from app.models import (
#     FeedbackRequest,
#     QueryRequest,
#     QueryResponse,
# )
# from app.rag.ingest import (
#     SUPPORTED,
#     ingest_path,
# )
# from app.services.feedback import (
#     save_feedback,
# )
# from app.services.orchestrator import (
#     VoiceRAGOrchestrator,
# )


# router = APIRouter(
#     prefix="/api/v1"
# )


# _orchestrator: (
#     VoiceRAGOrchestrator | None
# ) = None


# MAX_AUDIO_BYTES = (
#     25 * 1024 * 1024
# )


# # ============================================================
# # Spoken submit commands
# #
# # Important:
# # Groq Whisper is still batch STT.
# #
# # Therefore:
# #
# # "What is leave policy send"
# #       ↓
# # silence
# #       ↓
# # Whisper
# #       ↓
# # strip "send"
# #       ↓
# # "What is leave policy"
# #
# # Saying "send" does NOT currently trigger before silence.
# # ============================================================

# VOICE_SUBMIT_SUFFIX_PATTERNS = [

#     re.compile(
#         r"\s+(?:send|send it|submit|done|go)"
#         r"\s*[.!?]*$",
#         re.IGNORECASE,
#     ),

#     re.compile(
#         r"\s+(?:भेजो|भेज दो|भेज दें|हो गया|बस)"
#         r"\s*[।.!?]*$"
#     ),
# ]


# # ============================================================
# # Orchestrator singleton
# # ============================================================

# def orch() -> VoiceRAGOrchestrator:

#     global _orchestrator

#     if _orchestrator is None:

#         _orchestrator = (
#             VoiceRAGOrchestrator()
#         )

#     return _orchestrator


# # ============================================================
# # Language validation
# # ============================================================

# def validate_language(
#     language: str,
# ) -> str:

#     s = get_settings()

#     lang = (
#         language or
#         s.default_language
#     ).strip().lower()


#     if (
#         lang not in
#         s.supported_language_set
#     ):

#         allowed = ", ".join(
#             sorted(
#                 s.supported_language_set
#             )
#         )

#         raise HTTPException(
#             400,
#             (
#                 f"Unsupported language "
#                 f"'{lang}'. "
#                 f"Allowed: {allowed}"
#             ),
#         )


#     return lang


# # ============================================================
# # Remove trailing spoken control command
# # ============================================================

# def strip_voice_submit_suffix(
#     text: str,
# ) -> str:

#     cleaned = (
#         text or ""
#     ).strip()


#     for pattern in (
#         VOICE_SUBMIT_SUFFIX_PATTERNS
#     ):

#         candidate = pattern.sub(
#             "",
#             cleaned,
#         ).strip()


#         if (
#             candidate != cleaned
#         ):

#             return candidate


#     return cleaned


# # ============================================================
# # Determine filename from MIME
# # ============================================================

# def filename_for_mime(
#     mime_type: str,
# ) -> str:

#     mime = (
#         mime_type or ""
#     ).lower()


#     if "wav" in mime:

#         return "speech.wav"


#     if (
#         "mp4" in mime or
#         "m4a" in mime
#     ):

#         return "speech.mp4"


#     if "ogg" in mime:

#         return "speech.ogg"


#     return "speech.webm"


# # ============================================================
# # Health
# # ============================================================

# @router.get(
#     "/health"
# )
# async def health():

#     s = get_settings()


#     return {

#         "ok":
#             True,

#         "stt_provider":
#             s.stt_provider,

#         "llm_provider":
#             s.llm_provider,

#         "tts_provider":
#             s.tts_provider,

#         "embedding_model":
#             s.embedding_model,

#         "default_language":
#             s.default_language,

#         "supported_languages":
#             sorted(
#                 s.supported_language_set
#             ),
#     }


# # ============================================================
# # Text RAG
# # ============================================================

# @router.post(
#     "/query",
#     response_model=QueryResponse,
# )
# async def query(
#     req: QueryRequest,
# ):

#     language = validate_language(
#         req.language
#     )


#     (
#         request_id,
#         answer,
#         contexts,
#         timings,

#     ) = await orch().answer_text(

#         req.query,

#         language,

#         req.top_k,
#     )


#     return {

#         "request_id":
#             request_id,

#         "transcript":
#             None,

#         "answer":
#             answer,

#         "sources":
#             contexts,

#         "audio_base64":
#             None,

#         "audio_mime":
#             None,

#         "browser_tts_fallback":
#             False,

#         "timings_ms": {

#             k:
#                 round(v, 2)

#             for k, v
#             in timings.items()
#         },
#     }


# # ============================================================
# # Existing REST voice endpoint
# #
# # KEEP THIS.
# #
# # Useful for:
# # Swagger
# # debugging
# # fallback
# # non-WebSocket clients
# # ============================================================

# @router.post(
#     "/voice-query",
#     response_model=QueryResponse,
# )
# async def voice_query(

#     audio:
#         UploadFile =
#         File(...),

#     language:
#         str =
#         Form("en"),

#     top_k:
#         int =
#         Form(5),
# ):

#     language = validate_language(
#         language
#     )


#     body = await audio.read()


#     if not body:

#         raise HTTPException(
#             400,
#             "Empty audio file",
#         )


#     if (
#         len(body) >
#         MAX_AUDIO_BYTES
#     ):

#         raise HTTPException(
#             413,
#             (
#                 "Audio exceeds "
#                 "25 MB starter limit"
#             ),
#         )


#     return await orch().answer_voice(

#         audio.filename or
#         "speech.webm",

#         body,

#         audio.content_type or
#         "audio/webm",

#         language,

#         top_k,
#     )


# # ============================================================
# # Document upload
# # ============================================================

# @router.post(
#     "/documents"
# )
# async def upload_document(

#     file:
#         UploadFile =
#         File(...),
# ):

#     s = get_settings()


#     suffix = Path(
#         file.filename or ""
#     ).suffix.lower()


#     if (
#         suffix not in SUPPORTED
#     ):

#         raise HTTPException(

#             415,

#             (
#                 f"Unsupported file type "
#                 f"{suffix}; "
#                 f"use {sorted(SUPPORTED)}"
#             ),
#         )


#     data = await file.read()


#     if (
#         len(data) >
#         50 * 1024 * 1024
#     ):

#         raise HTTPException(

#             413,

#             (
#                 "Document exceeds "
#                 "50 MB starter limit"
#             ),
#         )


#     safe_name = Path(
#         file.filename
#     ).name


#     dest = (
#         s.docs_path /
#         safe_name
#     )


#     dest.write_bytes(
#         data
#     )


#     stats = ingest_path(
#         dest
#     )


#     return {

#         "stored_as":
#             safe_name,

#         **stats,
#     }


# # ============================================================
# # Feedback
# # ============================================================

# @router.post(
#     "/feedback"
# )
# async def feedback(
#     req: FeedbackRequest,
# ):

#     save_feedback(
#         req
#     )

#     return {
#         "ok":
#             True
#     }


# # ============================================================
# # CONTINUOUS VOICE WEBSOCKET
# # ============================================================

# @router.websocket(
#     "/ws/voice"
# )
# async def voice_ws(
#     websocket: WebSocket,
# ):

#     """

#     Phase-1 conversational voice socket.

#     Browser sends:

#     1. JSON metadata:

#        {
#          "type": "utterance.start",
#          "language": "en",
#          "top_k": 5,
#          "mime_type": "audio/wav"
#        }

#     2. Next WebSocket frame:

#        binary WAV audio


#     Browser performs:

#         microphone
#         +
#         VAD
#         +
#         silence detection


#     Backend performs:

#         STT
#         ↓
#         RAG
#         ↓
#         Qwen
#         ↓
#         TTS


#     Groq Whisper itself is still called
#     using its normal batch transcription API.

#     """


#     await websocket.accept()


#     await websocket.send_json({

#         "type":
#             "session.ready"
#     })


#     while True:

#         try:

#             # =================================================
#             # 1. Receive utterance metadata
#             # =================================================

#             metadata = (
#                 await websocket
#                 .receive_json()
#             )


#             if (
#                 metadata.get("type") !=
#                 "utterance.start"
#             ):

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         (
#                             "Expected an "
#                             "utterance.start "
#                             "metadata message."
#                         ),
#                 })

#                 continue


#             # =================================================
#             # Language
#             # =================================================

#             try:

#                 language = (
#                     validate_language(

#                         metadata.get(
#                             "language",
#                             "en",
#                         )
#                     )
#                 )

#             except HTTPException as exc:

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         str(exc.detail),
#                 })

#                 continue


#             # =================================================
#             # top_k validation
#             # =================================================

#             try:

#                 top_k = int(

#                     metadata.get(
#                         "top_k",
#                         5,
#                     )
#                 )

#             except (
#                 TypeError,
#                 ValueError,
#             ):

#                 top_k = 5


#             top_k = max(
#                 1,
#                 min(
#                     top_k,
#                     20,
#                 ),
#             )


#             # =================================================
#             # MIME
#             # =================================================

#             mime_type = str(

#                 metadata.get(
#                     "mime_type"
#                 ) or
#                 "audio/wav"
#             )


#             # =================================================
#             # 2. Receive binary utterance
#             # =================================================

#             audio_bytes = (
#                 await websocket
#                 .receive_bytes()
#             )


#             if not audio_bytes:

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         "Received empty audio.",
#                 })

#                 continue


#             if (
#                 len(audio_bytes) >
#                 MAX_AUDIO_BYTES
#             ):

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         (
#                             "Audio exceeds "
#                             "25 MB starter limit."
#                         ),
#                 })

#                 continue


#             total_started = (
#                 time.perf_counter()
#             )


#             timings: dict[
#                 str,
#                 float
#             ] = {}


#             # =================================================
#             # 3. STT
#             # =================================================

#             await websocket.send_json({

#                 "type":
#                     "stt.started"
#             })


#             stage_started = (
#                 time.perf_counter()
#             )


#             raw_transcript = (
#                 await orch()
#                 .stt
#                 .transcribe(

#                     filename_for_mime(
#                         mime_type
#                     ),

#                     audio_bytes,

#                     mime_type,

#                     language,
#                 )
#             )


#             timings["stt"] = (

#                 time.perf_counter() -
#                 stage_started

#             ) * 1000


#             # =================================================
#             # Remove optional trailing:
#             #
#             # send
#             # done
#             # go
#             # भेजो
#             # etc.
#             # =================================================

#             transcript = (
#                 strip_voice_submit_suffix(
#                     raw_transcript
#                 )
#             )


#             if not transcript:

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         (
#                             "No query remained "
#                             "after transcription."
#                         ),
#                 })

#                 continue


#             # Send transcript immediately.

#             await websocket.send_json({

#                 "type":
#                     "transcript",

#                 "text":
#                     transcript,

#                 "raw_text":
#                     raw_transcript,
#             })


#             # =================================================
#             # 4. RAG + LLM
#             #
#             # Reusing your EXISTING answer_text().
#             #
#             # Therefore document retrieval logic remains
#             # exactly the same as before.
#             # =================================================

#             await websocket.send_json({

#                 "type":
#                     "rag.started"
#             })


#             (
#                 request_id,
#                 answer,
#                 contexts,
#                 downstream,

#             ) = await orch().answer_text(

#                 transcript,

#                 language,

#                 top_k,
#             )


#             timings.update(
#                 downstream
#             )


#             # Send answer text before TTS completes.

#             await websocket.send_json({

#                 "type":
#                     "answer",

#                 "transcript":
#                     transcript,

#                 "text":
#                     answer,

#                 "sources":
#                     contexts,

#                 "timings_ms": {

#                     k:
#                         round(v, 2)

#                     for k, v
#                     in timings.items()
#                 },
#             })


#             # =================================================
#             # 5. TTS
#             # =================================================

#             await websocket.send_json({

#                 "type":
#                     "tts.started"
#             })


#             stage_started = (
#                 time.perf_counter()
#             )


#             browser_fallback = False

#             audio_b64 = None

#             audio_mime = None


#             try:

#                 spoken = (
#                     await orch()
#                     .tts
#                     .synthesize(

#                         answer,

#                         language,
#                     )
#                 )


#                 if spoken.data:

#                     audio_b64 = (
#                         base64
#                         .b64encode(
#                             spoken.data
#                         )
#                         .decode(
#                             "ascii"
#                         )
#                     )


#                     audio_mime = (
#                         spoken.mime_type
#                     )


#                 else:

#                     browser_fallback = (
#                         True
#                     )


#             except Exception:

#                 browser_fallback = (
#                     True
#                 )


#             timings["tts"] = (

#                 time.perf_counter() -
#                 stage_started

#             ) * 1000


#             timings["total"] = (

#                 time.perf_counter() -
#                 total_started

#             ) * 1000


#             # =================================================
#             # Complete final result
#             # =================================================

#             result = {

#                 "request_id":
#                     request_id,

#                 "transcript":
#                     transcript,

#                 "answer":
#                     answer,

#                 "sources":
#                     contexts,

#                 "audio_base64":
#                     audio_b64,

#                 "audio_mime":
#                     audio_mime,

#                 "browser_tts_fallback":
#                     browser_fallback,

#                 "timings_ms": {

#                     k:
#                         round(v, 2)

#                     for k, v
#                     in timings.items()
#                 },
#             }


#             await websocket.send_json({

#                 "type":
#                     "turn.result",

#                 "data":
#                     result,
#             })


#         # =====================================================
#         # User closed tab/session
#         # =====================================================

#         except WebSocketDisconnect:

#             break


#         # =====================================================
#         # Keep socket alive on an individual turn error
#         # =====================================================

#         except Exception as exc:

#             try:

#                 await websocket.send_json({

#                     "type":
#                         "error",

#                     "message":
#                         (
#                             "Voice turn failed: "
#                             f"{exc}"
#                         ),
#                 })

#             except Exception:

#                 break

from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.core.config import get_settings
from app.models import QueryRequest, QueryResponse, FeedbackRequest
from app.rag.ingest import ingest_path, SUPPORTED
from app.services.feedback import save_feedback
from app.services.orchestrator import VoiceRAGOrchestrator

router = APIRouter(prefix="/api/v1")
_orchestrator: VoiceRAGOrchestrator | None = None


def orch() -> VoiceRAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VoiceRAGOrchestrator()
    return _orchestrator


def validate_language(language: str) -> str:
    s = get_settings()
    lang = (language or s.default_language).strip().lower()
    if lang not in s.supported_language_set:
        allowed = ", ".join(sorted(s.supported_language_set))
        raise HTTPException(400, f"Unsupported language '{lang}'. Allowed: {allowed}")
    return lang


@router.get("/health")
async def health():
    s = get_settings()
    return {
        "ok": True,
        "stt_provider": s.stt_provider,
        "llm_provider": s.llm_provider,
        "tts_provider": s.tts_provider,
        "embedding_model": s.embedding_model,
        "default_language": s.default_language,
        "supported_languages": sorted(s.supported_language_set),
    }


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    language = validate_language(req.language)
    request_id, answer, contexts, timings = await orch().answer_text(req.query, language, req.top_k)
    return {
        "request_id": request_id,
        "transcript": None,
        "answer": answer,
        "sources": contexts,
        "audio_base64": None,
        "audio_mime": None,
        "browser_tts_fallback": False,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
    }


@router.post("/voice-query", response_model=QueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    top_k: int = Form(5),
):
    language = validate_language(language)
    body = await audio.read()
    if not body:
        raise HTTPException(400, "Empty audio file")
    if len(body) > 25 * 1024 * 1024:
        raise HTTPException(413, "Audio exceeds 25 MB starter limit")
    return await orch().answer_voice(
        audio.filename or "speech.webm",
        body,
        audio.content_type or "audio/webm",
        language,
        top_k,
    )


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    s = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(415, f"Unsupported file type {suffix}; use {sorted(SUPPORTED)}")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Document exceeds 50 MB starter limit")
    safe_name = Path(file.filename).name
    dest = s.docs_path / safe_name
    dest.write_bytes(data)
    stats = ingest_path(dest)
    return {"stored_as": safe_name, **stats}


@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    save_feedback(req)
    return {"ok": True}
