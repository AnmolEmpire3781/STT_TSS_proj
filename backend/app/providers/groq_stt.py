import httpx
from .base import STTProvider

class GroqSTT(STTProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required when STT_PROVIDER=groq")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def transcribe(self, filename: str, audio: bytes, mime_type: str, language: str | None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if language and language not in {"auto", ""}:
            data["language"] = language
        files = {"file": (filename, audio, mime_type or "application/octet-stream")}
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            r.raise_for_status()
            text = r.json().get("text", "").strip()
        if not text:
            raise RuntimeError("STT returned empty transcription")
        return text
