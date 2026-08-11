# import base64
# import time
# import uuid
# from app.core.config import get_settings
# from app.providers.factory import build_stt, build_llm, build_tts
# from app.rag.vector_store import get_vector_store


# class VoiceRAGOrchestrator:
#     def __init__(self):
#         self.s = get_settings()
#         self.stt = build_stt(self.s)
#         self.llm = build_llm(self.s)
#         self.tts = build_tts(self.s)

#     async def answer_text(self, query: str, language: str = "en", top_k: int | None = None):
#         request_id = str(uuid.uuid4())
#         timings = {}

#         t = time.perf_counter()
#         contexts = get_vector_store().query(
#             query,
#             top_k or self.s.rag_top_k,
#             self.s.rag_min_score,
#         )
#         timings["retrieval"] = (time.perf_counter() - t) * 1000

#         t = time.perf_counter()
#         answer = await self.llm.answer(query, contexts, language)
#         timings["llm"] = (time.perf_counter() - t) * 1000

#         return request_id, answer, contexts, timings

#     async def answer_voice(
#         self,
#         filename: str,
#         audio: bytes,
#         mime_type: str,
#         language: str,
#         top_k: int | None = None,
#     ):
#         total = time.perf_counter()
#         timings = {}

#         t = time.perf_counter()
#         transcript = await self.stt.transcribe(filename, audio, mime_type, language)
#         timings["stt"] = (time.perf_counter() - t) * 1000

#         request_id, answer, contexts, downstream = await self.answer_text(transcript, language, top_k)
#         timings.update(downstream)

#         t = time.perf_counter()
#         browser_fallback = False
#         audio_b64 = None
#         audio_mime = None
#         try:
#             spoken = await self.tts.synthesize(answer, language)
#             if spoken.data:
#                 audio_b64 = base64.b64encode(spoken.data).decode("ascii")
#                 audio_mime = spoken.mime_type
#             else:
#                 browser_fallback = True
#         except Exception:
#             browser_fallback = True
#         timings["tts"] = (time.perf_counter() - t) * 1000
#         timings["total"] = (time.perf_counter() - total) * 1000

#         return {
#             "request_id": request_id,
#             "transcript": transcript,
#             "answer": answer,
#             "sources": contexts,
#             "audio_base64": audio_b64,
#             "audio_mime": audio_mime,
#             "browser_tts_fallback": browser_fallback,
#             "timings_ms": {k: round(v, 2) for k, v in timings.items()},
#         }

import base64
import time
import uuid

from app.core.config import get_settings
from app.providers.factory import (
    build_stt,
    build_llm,
    build_tts,
)
from app.rag.vector_store import get_vector_store


class VoiceRAGOrchestrator:

    def __init__(self):

        self.s = get_settings()

        self.stt = build_stt(
            self.s
        )

        self.llm = build_llm(
            self.s
        )

        self.tts = build_tts(
            self.s
        )


    # ========================================================
    # TEXT PIPELINE
    # ========================================================

    async def answer_text(
        self,
        query: str,
        language: str = "en",
        top_k: int | None = None,
        rag_based: bool = True,
    ):

        request_id = str(
            uuid.uuid4()
        )

        timings = {}

        total_started = (
            time.perf_counter()
        )


        # ====================================================
        # MODE 1: DOCUMENT RAG
        # ====================================================

        if rag_based:

            retrieval_started = (
                time.perf_counter()
            )


            contexts = (
                get_vector_store().query(
                    query,
                    top_k
                    or self.s.rag_top_k,
                    self.s.rag_min_score,
                )
            )


            timings["retrieval"] = (
                (
                    time.perf_counter()
                    - retrieval_started
                )
                * 1000
            )


        # ====================================================
        # MODE 2: DIRECT LLM
        #
        # Do NOT initialize/query vector DB.
        # ====================================================

        else:

            contexts = []

            timings["retrieval"] = 0.0


        # ====================================================
        # LLM
        # ====================================================

        llm_started = (
            time.perf_counter()
        )


        answer = await self.llm.answer(
            query,
            contexts,
            language,
            rag_based,
        )


        timings["llm"] = (
            (
                time.perf_counter()
                - llm_started
            )
            * 1000
        )


        timings["total"] = (
            (
                time.perf_counter()
                - total_started
            )
            * 1000
        )


        return (
            request_id,
            answer,
            contexts,
            timings,
        )


    # ========================================================
    # VOICE PIPELINE
    #
    # Stable flow remains:
    #
    # Record
    #   ↓
    # Stop
    #   ↓
    # Whisper
    #   ↓
    # RAG or Direct LLM
    #   ↓
    # TTS
    # ========================================================

    async def answer_voice(
        self,
        filename: str,
        audio: bytes,
        mime_type: str,
        language: str,
        top_k: int | None = None,
        rag_based: bool = True,
    ):

        total_started = (
            time.perf_counter()
        )

        timings = {}


        # ====================================================
        # STT
        # ====================================================

        stt_started = (
            time.perf_counter()
        )


        transcript = (
            await self.stt.transcribe(
                filename,
                audio,
                mime_type,
                language,
            )
        )


        timings["stt"] = (
            (
                time.perf_counter()
                - stt_started
            )
            * 1000
        )


        # ====================================================
        # RAG / DIRECT MODE
        # ====================================================

        (
            request_id,
            answer,
            contexts,
            downstream,

        ) = await self.answer_text(
            transcript,
            language,
            top_k,
            rag_based,
        )


        # Keep retrieval + LLM timings.
        #
        # answer_text() has its own total;
        # final voice total will overwrite it below.
        timings.update(
            downstream
        )


        # ====================================================
        # TTS
        # ====================================================

        tts_started = (
            time.perf_counter()
        )


        browser_fallback = False

        audio_b64 = None

        audio_mime = None


        try:

            spoken = (
                await self.tts.synthesize(
                    answer,
                    language,
                )
            )


            if spoken.data:

                audio_b64 = (
                    base64
                    .b64encode(
                        spoken.data
                    )
                    .decode(
                        "ascii"
                    )
                )


                audio_mime = (
                    spoken.mime_type
                )


            else:

                browser_fallback = True


        except Exception:

            browser_fallback = True


        timings["tts"] = (
            (
                time.perf_counter()
                - tts_started
            )
            * 1000
        )


        timings["total"] = (
            (
                time.perf_counter()
                - total_started
            )
            * 1000
        )


        # ====================================================
        # Response
        # ====================================================

        return {
            "request_id":
                request_id,

            "transcript":
                transcript,

            "answer":
                answer,

            "sources":
                contexts,

            "audio_base64":
                audio_b64,

            "audio_mime":
                audio_mime,

            "browser_tts_fallback":
                browser_fallback,

            "timings_ms": {
                k: round(v, 2)
                for k, v
                in timings.items()
            },

            "rag_based":
                rag_based,
        }