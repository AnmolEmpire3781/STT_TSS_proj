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
import re
import time
import uuid

from app.core.config import get_settings
from app.providers.factory import (
    build_stt,
    build_llm,
    build_tts,
)
from app.rag.vector_store import get_vector_store


# Splits on sentence-ending punctuation (incl. Hindi danda) followed by whitespace.
SENTENCE_END_RE = re.compile(r"[.!?।]+\s+")


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


    # ========================================================
    # STREAMING TEXT PIPELINE
    #
    # Retrieves once, then streams LLM tokens. As soon as a full
    # sentence has arrived it is synthesized to audio immediately,
    # instead of waiting for the whole answer, cutting perceived
    # latency. Yields dict events; the last event has type "final".
    # ========================================================

    async def answer_text_streaming(
        self,
        query: str,
        language: str = "en",
        top_k: int | None = None,
        rag_based: bool = True,
    ):
        request_id = str(uuid.uuid4())
        timings = {}
        total_started = time.perf_counter()

        if rag_based:
            retrieval_started = time.perf_counter()
            contexts = get_vector_store().query(
                query,
                top_k or self.s.rag_top_k,
                self.s.rag_min_score,
            )
            timings["retrieval"] = (time.perf_counter() - retrieval_started) * 1000
        else:
            contexts = []
            timings["retrieval"] = 0.0

        yield {"type": "contexts", "request_id": request_id, "contexts": contexts}

        llm_started = time.perf_counter()
        buffer = ""
        full_answer = ""
        sentence_index = 0

        async for delta in self.llm.answer_stream(query, contexts, language, rag_based):
            full_answer += delta
            buffer += delta
            yield {"type": "delta", "text": delta}

            while True:
                match = SENTENCE_END_RE.search(buffer)
                if not match:
                    break

                sentence = buffer[: match.end()].strip()
                buffer = buffer[match.end() :]

                if sentence:
                    async for event in self._synthesize_sentence(sentence, language, sentence_index):
                        yield event
                    sentence_index += 1

        # Flush trailing text that had no closing punctuation.
        if buffer.strip():
            async for event in self._synthesize_sentence(buffer.strip(), language, sentence_index):
                yield event
            sentence_index += 1

        timings["llm"] = (time.perf_counter() - llm_started) * 1000
        timings["total"] = (time.perf_counter() - total_started) * 1000

        yield {
            "type": "final",
            "request_id": request_id,
            "answer": full_answer.strip(),
            "sources": contexts,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "rag_based": rag_based,
        }

    async def _synthesize_sentence(self, sentence: str, language: str, index: int):
        try:
            spoken = await self.tts.synthesize(sentence, language)
        except Exception:
            spoken = None

        audio_b64 = None
        audio_mime = None
        if spoken is not None and spoken.data:
            audio_b64 = base64.b64encode(spoken.data).decode("ascii")
            audio_mime = spoken.mime_type

        yield {
            "type": "sentence_audio",
            "index": index,
            "text": sentence,
            "audio_base64": audio_b64,
            "audio_mime": audio_mime,
        }