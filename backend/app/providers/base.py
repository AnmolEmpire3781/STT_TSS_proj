# from dataclasses import dataclass
# from typing import Protocol


# @dataclass
# class AudioResult:
#     data: bytes | None
#     mime_type: str | None


# class STTProvider(Protocol):
#     async def transcribe(self, filename: str, audio: bytes, mime_type: str, language: str | None) -> str: ...


# class LLMProvider(Protocol):
#     async def answer(self, question: str, contexts: list[dict], language: str) -> str: ...


# class TTSProvider(Protocol):
#     async def synthesize(self, text: str, language: str | None) -> AudioResult: ...

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AudioResult:
    data: bytes | None
    mime_type: str | None


class STTProvider(Protocol):

    async def transcribe(
        self,
        filename: str,
        audio: bytes,
        mime_type: str,
        language: str | None,
    ) -> str:
        ...


class LLMProvider(Protocol):

    async def answer(
        self,
        question: str,
        contexts: list[dict],
        language: str,
        rag_based: bool = True,
    ) -> str:
        ...


class TTSProvider(Protocol):

    async def synthesize(
        self,
        text: str,
        language: str | None,
    ) -> AudioResult:
        ...