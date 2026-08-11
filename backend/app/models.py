# from typing import Any, Literal
# from pydantic import BaseModel, Field

# LanguageCode = Literal["en", "hi"]


# class SourceChunk(BaseModel):
#     id: str
#     source: str
#     page: int | None = None
#     score: float
#     text: str


# class QueryRequest(BaseModel):
#     query: str = Field(min_length=1, max_length=8000)
#     language: LanguageCode = "en"
#     top_k: int | None = Field(default=None, ge=1, le=20)


# class QueryResponse(BaseModel):
#     request_id: str
#     transcript: str | None = None
#     answer: str
#     sources: list[SourceChunk]
#     audio_base64: str | None = None
#     audio_mime: str | None = None
#     browser_tts_fallback: bool = False
#     timings_ms: dict[str, float]


# class FeedbackRequest(BaseModel):
#     request_id: str
#     language: LanguageCode = "en"
#     rating: int = Field(ge=-1, le=1)
#     query: str
#     model_answer: str
#     corrected_answer: str | None = None
#     note: str | None = None
#     sources: list[dict[str, Any]] = Field(default_factory=list)

from typing import Any, Literal

from pydantic import BaseModel, Field


LanguageCode = Literal["en", "hi"]


class SourceChunk(BaseModel):
    id: str
    source: str
    page: int | None = None
    score: float
    text: str


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=8000,
    )

    language: LanguageCode = "en"

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )

    # ---------------------------------------------------------
    # Answer mode
    #
    # True:
    # Query -> BGE-M3 -> Chroma -> Context -> LLM
    #
    # False:
    # Query -> LLM directly
    #
    # Default True preserves our original application behavior.
    # ---------------------------------------------------------
    rag_based: bool = True


class QueryResponse(BaseModel):
    request_id: str

    transcript: str | None = None

    answer: str

    sources: list[SourceChunk]

    audio_base64: str | None = None

    audio_mime: str | None = None

    browser_tts_fallback: bool = False

    timings_ms: dict[str, float]

    # Allows frontend/feedback/evaluation to know
    # which pipeline generated this response.
    rag_based: bool = True


class FeedbackRequest(BaseModel):
    request_id: str

    language: LanguageCode = "en"

    rating: int = Field(
        ge=-1,
        le=1,
    )

    query: str

    model_answer: str

    corrected_answer: str | None = None

    note: str | None = None

    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )

    # Important for later comparison:
    #
    # RAG positive rate vs Direct-LLM positive rate.
    rag_based: bool = True