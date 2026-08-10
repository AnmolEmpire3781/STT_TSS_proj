from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_DIR / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"

    default_language: str = "en"
    supported_languages: str = "en,hi"

    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_llm_model: str = "qwen/qwen3.6-27b"
    groq_llm_reasoning_effort: str = "none"

    stt_provider: str = "groq"
    llm_provider: str = "groq"
    tts_provider: str = "edge"

    embedding_model: str = "BAAI/bge-m3"
    chroma_collection: str = "knowledge"
    rag_top_k: int = 5
    rag_min_score: float = 0.20
    chunk_size: int = 700
    chunk_overlap: int = 120

    edge_tts_voice_en: str = "en-IN-NeerjaNeural"
    edge_tts_voice_hi: str = "hi-IN-SwaraNeural"

    local_llm_base_url: str = "http://localhost:8001/v1"
    local_llm_api_key: str = "local-not-secret"
    local_llm_model: str = "qwen-domain"
    local_asr_url: str = "http://localhost:8002/transcribe"
    local_tts_url: str = "http://localhost:8003/synthesize"
    local_tts_voice_en: str = "Thoma"
    local_tts_voice_hi: str = "Divya"

    @property
    def supported_language_set(self) -> set[str]:
        return {x.strip().lower() for x in self.supported_languages.split(",") if x.strip()}

    @property
    def chroma_path(self) -> Path:
        p = BACKEND_DIR / "data" / "chroma"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def docs_path(self) -> Path:
        p = BACKEND_DIR / "data" / "docs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def feedback_path(self) -> Path:
        p = BACKEND_DIR / "data" / "feedback" / "feedback.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
