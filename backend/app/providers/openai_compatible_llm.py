# import httpx

# from .base import LLMProvider
# from app.services.text_cleanup import clean_assistant_answer

# SYSTEM_PROMPT = """
# You are an intelligent document-grounded voice assistant.

# Your job is to answer the user's question naturally using the
# retrieved document context.

# GROUNDING RULES:

# 1. Use only the supplied context for organization-specific,
#    document-specific, or domain-specific factual claims.

# 2. Never invent facts that are not supported by the supplied context.

# 3. If the supplied context does not contain enough information,
#    clearly say that the information is not available in the
#    provided knowledge base.

# 4. Do not mention source IDs such as S1, S2, S3, S4, etc.
#    Source metadata is displayed separately by the application.

# LANGUAGE RULES:

# 5. If requested language is English, answer in clear natural English.

# 6. If requested language is Hindi, answer in natural conversational
#    Hindi using Devanagari script.

# 7. In Hindi answers, preserve commonly used English technical terms,
#    product names, APIs, model names, acronyms, and proper nouns when
#    that sounds more natural.

# 8. Reason directly over English, Hindi, or mixed-language documents.
#    Do not unnecessarily translate documents before answering.

# VOICE RESPONSE RULES:

# 9. The answer will be spoken aloud by a text-to-speech engine.

# 10. Produce plain conversational text only.

# 11. Never use Markdown formatting.

# 12. Never use:
#     - markdown bold markers such as **
#     - markdown italic markers such as *
#     - markdown headings such as #
#     - markdown bullet symbols
#     - citation markers such as [S1], [S2]
#     - LaTeX delimiters such as $
#     - LaTeX commands
#     - code fences
#     - raw mathematical notation when it can be explained naturally

# 13. Convert mathematical or technical notation into natural spoken
#     language whenever possible.

# Examples:

# Instead of:
# "The dimension $d_{model}$ is 512 [S4]."

# Say:
# "The model dimension is 512."

# Instead of:
# "LayerNorm(x + Sublayer(x))"

# Say:
# "Layer normalization is applied after adding the sublayer output
# to the original input."

# Instead of:
# "**Architecture:** The model uses attention."

# Say:
# "The architecture uses attention."

# 14. Do not begin answers with repetitive phrases such as
#     "Based on the provided context" unless that qualification is
#     actually important.

# 15. Answer the user's actual question directly.

# 16. Prefer short paragraphs and natural spoken sentences.

# 17. Be concise, but preserve important facts, conditions, numbers,
#     limitations, and steps.

# 18. Do not read out document identifiers, page numbers, citation
#     markers, formatting symbols, URLs, or metadata unless the user
#     explicitly asks for them.
# 19. Answer only what the user asked.

# 20. Do not summarize every retrieved passage.

# 21. Retrieved passages are evidence, not a checklist of facts that
#     must all appear in the answer.

# 22. Select only the facts necessary to answer the question.

# 23. If the user's question asks for a basic explanation, explain the
#     concept first in simple language before giving technical details.

# 24. Do not include benchmark scores, training hardware, dates,
#     historical details, equations, or implementation specifics
#     unless they directly help answer the question or the user asks
#     for them.
# """

# LANGUAGE_LABEL = {"en": "English", "hi": "Hindi"}


# class OpenAICompatibleLLM(LLMProvider):
#     def __init__(self, api_key: str, base_url: str, model: str, reasoning_effort: str | None = None):
#         self.api_key = api_key
#         self.base_url = base_url.rstrip("/")
#         self.model = model
#         self.reasoning_effort = reasoning_effort

#     async def answer(self, question: str, contexts: list[dict], language: str = "en") -> str:
#         if contexts:
#             context_text = "\n\n".join(
#                 f"[S{i}] source={c['source']} page={c.get('page')}\n{c['text']}"
#                 for i, c in enumerate(contexts, 1)
#             )
#         else:
#             context_text = "(No relevant context retrieved.)"

#         requested_language = LANGUAGE_LABEL.get(language, "English")
#         user = (
#             f"Requested answer language: {requested_language} ({language})\n\n"
#             f"Context:\n{context_text}\n\n"
#             f"Question:\n{question}"
#         )
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user},
#             ],
#             "temperature": 0.2,
#             "max_tokens": 400,
#         }
#         if self.reasoning_effort:
#             payload["reasoning_effort"] = self.reasoning_effort

#         headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
#         async with httpx.AsyncClient(timeout=120) as client:
#             r = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
#             r.raise_for_status()
#             obj = r.json()
#         # answer = obj["choices"][0]["message"]["content"].strip()
#         # if not answer:
#         #     raise RuntimeError("LLM returned an empty answer")
#         # return answer
        
#         raw_answer = (
#             obj["choices"][0]["message"]["content"]
#             .strip()
#         )


#         if not raw_answer:

#             raise RuntimeError(
#                 "LLM returned an empty answer"
#             )


#         answer = clean_assistant_answer(
#             raw_answer
#         )


#         if not answer:

#             raise RuntimeError(
#                 "LLM answer became empty after cleanup"
#             )


#         return answer


import json

import httpx

from .base import LLMProvider
from app.services.text_cleanup import clean_assistant_answer


# ============================================================
# RAG MODE PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are an intelligent document-grounded voice assistant.

Your job is to answer the user's question naturally using the
retrieved document context.

GROUNDING RULES:

1. Use only the supplied context for organization-specific,
   document-specific, or domain-specific factual claims.

2. Never invent facts that are not supported by the supplied context.

3. If the supplied context does not contain enough information,
   clearly say that the information is not available in the
   provided knowledge base.

4. Do not mention source IDs such as S1, S2, S3, S4, etc.
   Source metadata is displayed separately by the application.

LANGUAGE RULES:

5. If requested language is English, answer in clear natural English.

6. If requested language is Hindi, answer in natural conversational
   Hindi using Devanagari script.

7. The user may use English, Hindi, Hinglish, or mixed technical
   terminology.

8. In Hindi answers, preserve commonly used English technical terms,
   product names, APIs, model names, acronyms, and proper nouns when
   that sounds more natural.

9. Reason directly over English, Hindi, or mixed-language documents.
   Do not unnecessarily translate documents before answering.

VOICE RESPONSE RULES:

10. The answer will be spoken aloud by a text-to-speech engine.

11. Produce plain conversational text only.

12. Never use Markdown formatting.

13. Never use:
    - markdown bold markers
    - markdown italic markers
    - markdown headings
    - markdown bullet symbols
    - citation markers such as [S1], [S2]
    - LaTeX delimiters
    - LaTeX commands
    - code fences
    - raw mathematical notation when it can be explained naturally

14. Convert mathematical or technical notation into natural spoken
    language whenever possible.

Example:

Instead of:
"The dimension d_model is 512 [S4]."

Say:
"The model dimension is 512."

Instead of:
"LayerNorm(x + Sublayer(x))"

Say:
"Layer normalization is applied after adding the sublayer output
to the original input."

15. Do not begin answers with repetitive phrases such as
    "Based on the provided context" unless that qualification is
    actually important.

16. Answer the user's actual question directly.

17. Prefer short paragraphs and natural spoken sentences.

18. Be concise, but preserve important facts, conditions, numbers,
    limitations, and steps.

19. Do not read out document identifiers, page numbers, citation
    markers, formatting symbols, URLs, or metadata unless the user
    explicitly asks for them.

20. Do not summarize every retrieved passage.

21. Retrieved passages are evidence, not a checklist of facts that
    must all appear in the answer.

22. Select only the facts necessary to answer the question.

23. If the user's question asks for a basic explanation, explain the
    concept first in simple language before giving technical details.

24. Do not include benchmark scores, training hardware, dates,
    historical details, equations, or implementation specifics
    unless they directly help answer the question or the user asks
    for them.
"""


# ============================================================
# DIRECT LLM MODE PROMPT
#
# IMPORTANT:
# No uploaded document context is available in this mode.
# ============================================================

DIRECT_SYSTEM_PROMPT = """
You are an intelligent general-purpose voice assistant.

The application is currently operating in Direct LLM mode.

No private document retrieval or RAG context has been supplied to you.

ANSWERING RULES:

1. Answer the user's question using your general knowledge and
   reasoning.

2. Do not claim that you searched, retrieved, or read the user's
   uploaded documents.

3. Do not pretend to know private company information, internal
   policies, private research documents, or organization-specific
   facts that have not been supplied to you.

4. If the user asks for private or document-specific information,
   explain that Document RAG mode should be used rather than
   inventing the answer.

5. Clearly distinguish uncertainty from fact when necessary.

LANGUAGE RULES:

6. If requested language is English, answer in clear natural English.

7. If requested language is Hindi, answer in natural conversational
   Hindi using Devanagari script.

8. The user may speak English, Hindi, Hinglish, Romanized Hindi,
   or mixed technical language.

9. Understand the meaning of mixed-language queries naturally.

10. Preserve technical words, product names, APIs, model names,
    acronyms, and proper nouns in English when appropriate.

VOICE RESPONSE RULES:

11. The response will be spoken by a text-to-speech engine.

12. Produce plain conversational text.

13. Do not use Markdown, source IDs, citation markers, LaTeX,
    formatting symbols, code fences, or unnecessary URLs.

14. Convert technical notation into natural spoken language where
    practical.

15. Answer the user's actual question directly.

16. Be concise unless the user explicitly requests a detailed
    explanation.

17. Do not include unrelated facts merely because you know them.

18. Do not claim access to live/current information unless it was
    explicitly supplied in the conversation.
"""


LANGUAGE_LABEL = {
    "en": "English",
    "hi": "Hindi",
}


class OpenAICompatibleLLM(LLMProvider):

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort


    def _build_prompt(
        self,
        question: str,
        contexts: list[dict],
        language: str,
        rag_based: bool,
    ) -> tuple[str, str]:

        requested_language = LANGUAGE_LABEL.get(
            language,
            "English",
        )


        # ====================================================
        # RAG MODE
        # ====================================================

        if rag_based:

            if contexts:

                context_text = "\n\n".join(
                    (
                        f"[S{i}] "
                        f"source={c['source']} "
                        f"page={c.get('page')}\n"
                        f"{c['text']}"
                    )
                    for i, c in enumerate(
                        contexts,
                        1,
                    )
                )

            else:

                context_text = (
                    "(No relevant context retrieved.)"
                )


            system_prompt = RAG_SYSTEM_PROMPT


            user_prompt = (
                f"Requested answer language: "
                f"{requested_language} ({language})\n\n"
                f"Retrieved document context:\n"
                f"{context_text}\n\n"
                f"Question:\n"
                f"{question}"
            )


        # ====================================================
        # DIRECT LLM MODE
        #
        # IMPORTANT:
        #
        # No contexts are passed into the prompt.
        # No Chroma output is visible to Qwen.
        # ====================================================

        else:

            system_prompt = DIRECT_SYSTEM_PROMPT


            user_prompt = (
                f"Requested answer language: "
                f"{requested_language} ({language})\n\n"
                f"Question:\n"
                f"{question}"
            )


        return system_prompt, user_prompt


    async def answer(
        self,
        question: str,
        contexts: list[dict],
        language: str = "en",
        rag_based: bool = True,
    ) -> str:

        system_prompt, user_prompt = self._build_prompt(
            question, contexts, language, rag_based
        )

        # ====================================================
        # Groq/OpenAI-compatible request
        # ====================================================

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            "temperature": 0.2,

            "max_tokens": 400,
        }


        if self.reasoning_effort:

            payload["reasoning_effort"] = (
                self.reasoning_effort
            )


        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),

            "Content-Type": (
                "application/json"
            ),
        }


        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            response = await client.post(
                (
                    f"{self.base_url}"
                    f"/chat/completions"
                ),
                headers=headers,
                json=payload,
            )


            response.raise_for_status()


            obj = response.json()


        # ====================================================
        # Extract model answer
        # ====================================================

        raw_answer = (
            obj["choices"][0]
            ["message"]
            ["content"]
            .strip()
        )


        if not raw_answer:

            raise RuntimeError(
                "LLM returned an empty answer"
            )


        # ====================================================
        # Deterministic TTS/UI cleanup
        #
        # Removes:
        # Markdown
        # LaTeX
        # source markers
        # unnecessary formatting
        # ====================================================

        answer = clean_assistant_answer(
            raw_answer
        )


        if not answer:

            raise RuntimeError(
                "LLM answer became empty after cleanup"
            )


        return answer


    # ============================================================
    # STREAMING
    #
    # Yields raw text deltas as they arrive over SSE, so the caller
    # can start TTS on completed sentences before the full answer
    # has finished generating (lowers perceived latency).
    #
    # NOTE: deltas are NOT passed through clean_assistant_answer();
    # callers should clean the final assembled text if needed.
    # ============================================================

    async def answer_stream(
        self,
        question: str,
        contexts: list[dict],
        language: str = "en",
        rag_based: bool = True,
    ):

        system_prompt, user_prompt = self._build_prompt(
            question, contexts, language, rag_based
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
            "stream": True,
        }

        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue

                    delta = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        yield delta