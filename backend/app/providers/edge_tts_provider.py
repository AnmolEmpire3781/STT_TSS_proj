import edge_tts
from .base import AudioResult, TTSProvider


class EdgeTTSProvider(TTSProvider):
    def __init__(self, voices: dict[str, str]):
        self.voices = voices

    async def synthesize(self, text: str, language: str | None) -> AudioResult:
        # Prototype-only convenience adapter: no API key, but no production SLA.
        lang = language if language in self.voices else "en"
        voice = self.voices[lang]
        communicator = edge_tts.Communicate(text=text, voice=voice)
        audio = bytearray()
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        return AudioResult(bytes(audio) if audio else None, "audio/mpeg" if audio else None)


class BrowserTTSProvider(TTSProvider):
    async def synthesize(self, text: str, language: str | None) -> AudioResult:
        return AudioResult(None, None)
