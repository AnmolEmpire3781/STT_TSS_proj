from app.core.config import Settings
from .groq_stt import GroqSTT
from .openai_compatible_llm import OpenAICompatibleLLM
from .edge_tts_provider import EdgeTTSProvider, BrowserTTSProvider
from .local_http import LocalASRHTTP, LocalTTSHTTP


def build_stt(s: Settings):
    if s.stt_provider == "groq":
        return GroqSTT(s.groq_api_key, s.groq_base_url, s.groq_stt_model)
    if s.stt_provider in {"faster_whisper", "local"}:
        return LocalASRHTTP(s.local_asr_url)
    raise ValueError(f"Unknown STT_PROVIDER={s.stt_provider}")


def build_llm(s: Settings):
    if s.llm_provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return OpenAICompatibleLLM(
            s.groq_api_key, s.groq_base_url, s.groq_llm_model, s.groq_llm_reasoning_effort
        )
    if s.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(
            s.local_llm_api_key, s.local_llm_base_url, s.local_llm_model, None
        )
    raise ValueError(f"Unknown LLM_PROVIDER={s.llm_provider}")


def build_tts(s: Settings):
    if s.tts_provider == "edge":
        return EdgeTTSProvider({"en": s.edge_tts_voice_en, "hi": s.edge_tts_voice_hi})
    if s.tts_provider == "browser":
        return BrowserTTSProvider()
    if s.tts_provider in {"indic_parler", "local"}:
        return LocalTTSHTTP(
            s.local_tts_url,
            {"en": s.local_tts_voice_en, "hi": s.local_tts_voice_hi},
        )
    raise ValueError(f"Unknown TTS_PROVIDER={s.tts_provider}")
