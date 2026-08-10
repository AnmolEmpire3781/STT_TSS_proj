import httpx
from .base import STTProvider, TTSProvider, AudioResult


class LocalASRHTTP(STTProvider):
    def __init__(self, url: str):
        self.url = url

    async def transcribe(self, filename: str, audio: bytes, mime_type: str, language: str | None) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                self.url,
                data={"language": language or "en"},
                files={"audio": (filename, audio, mime_type or "application/octet-stream")},
            )
            r.raise_for_status()
        text = r.json().get("text", "").strip()
        if not text:
            raise RuntimeError("Local ASR returned empty transcription")
        return text


class LocalTTSHTTP(TTSProvider):
    def __init__(self, url: str, voices: dict[str, str]):
        self.url = url
        self.voices = voices

    async def synthesize(self, text: str, language: str | None) -> AudioResult:
        lang = language if language in self.voices else "en"
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                self.url,
                json={"text": text, "language": lang, "voice": self.voices.get(lang)},
            )
            r.raise_for_status()
        return AudioResult(r.content, r.headers.get("content-type", "audio/wav"))
