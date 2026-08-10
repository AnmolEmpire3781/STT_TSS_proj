import html
import re


# Matches:
# [S1]
# [S2]
# [S1, S2]
# [S1; S4]
SOURCE_CITATION_RE = re.compile(
    r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]",
    re.IGNORECASE,
)


def clean_assistant_answer(text: str) -> str:
    """
    Convert an LLM response into clean, conversational text
    suitable for both UI display and text-to-speech.

    Removes:
    - RAG source markers such as [S1]
    - Markdown formatting
    - Markdown links
    - URLs
    - LaTeX delimiters
    - common LaTeX commands
    - unnecessary formatting characters
    """

    if not text:
        return ""

    value = html.unescape(text).strip()

    # ---------------------------------------------------------
    # Remove fenced-code markers
    # ---------------------------------------------------------
    value = re.sub(
        r"```(?:[a-zA-Z0-9_+\-]+)?",
        "",
        value,
    )

    value = value.replace("```", "")

    # ---------------------------------------------------------
    # Remove RAG source citations
    # [S1], [S2], [S1, S4], etc.
    # ---------------------------------------------------------
    value = SOURCE_CITATION_RE.sub(
        "",
        value,
    )

    # ---------------------------------------------------------
    # Markdown links:
    # [Transformer](https://example.com)
    # ->
    # Transformer
    # ---------------------------------------------------------
    value = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        value,
    )

    # ---------------------------------------------------------
    # Remove URLs
    # ---------------------------------------------------------
    value = re.sub(
        r"https?://\S+",
        "",
        value,
    )

    # ---------------------------------------------------------
    # Markdown headings
    # ### Architecture -> Architecture
    # ---------------------------------------------------------
    value = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s*",
        "",
        value,
    )

    # ---------------------------------------------------------
    # Markdown bullets
    # * Item
    # - Item
    # + Item
    # ->
    # Item
    # ---------------------------------------------------------
    value = re.sub(
        r"(?m)^\s*[-*+]\s+",
        "",
        value,
    )

    # ---------------------------------------------------------
    # Remove Markdown bold/italics markers
    # ---------------------------------------------------------
    value = value.replace("**", "")
    value = value.replace("__", "")

    value = re.sub(
        r"(?<!\w)[*_](?!\w)",
        "",
        value,
    )

    # ---------------------------------------------------------
    # Remove LaTeX delimiters
    # ---------------------------------------------------------
    value = value.replace("$$", "")
    value = value.replace("$", "")

    value = value.replace(r"\(", "")
    value = value.replace(r"\)", "")
    value = value.replace(r"\[", "")
    value = value.replace(r"\]", "")

    # ---------------------------------------------------------
    # Common LaTeX wrappers
    #
    # \text{hello} -> hello
    # \mathbf{hello} -> hello
    # ---------------------------------------------------------
    value = re.sub(
        r"\\(?:text|mathrm|mathbf|mathit|operatorname)"
        r"\{([^{}]*)\}",
        r"\1",
        value,
    )

    # ---------------------------------------------------------
    # Convert common mathematical commands into
    # speech-friendly text
    # ---------------------------------------------------------
    replacements = {
        r"\times": " times ",
        r"\cdot": " times ",
        r"\approx": " approximately ",
        r"\leq": " less than or equal to ",
        r"\geq": " greater than or equal to ",
        r"\neq": " not equal to ",
        r"\rightarrow": " leads to ",
        r"\to": " to ",
        r"\infty": " infinity ",
        r"\sum": " sum ",
        r"\sqrt": " square root ",
        r"\alpha": " alpha ",
        r"\beta": " beta ",
        r"\gamma": " gamma ",
    }

    for source, replacement in replacements.items():
        value = value.replace(
            source,
            replacement,
        )

    # ---------------------------------------------------------
    # d_{model} -> d model
    # d_{ff}    -> d ff
    # ---------------------------------------------------------
    value = re.sub(
        r"([A-Za-z0-9]+)_\{([^{}]+)\}",
        r"\1 \2",
        value,
    )

    # x_i -> x i
    value = re.sub(
        r"([A-Za-z0-9]+)_([A-Za-z0-9]+)",
        r"\1 \2",
        value,
    )

    # x^{2} -> x 2
    value = re.sub(
        r"\^\{([^{}]+)\}",
        r" \1",
        value,
    )

    # ---------------------------------------------------------
    # Remove remaining braces
    # ---------------------------------------------------------
    value = value.replace("{", "")
    value = value.replace("}", "")

    # ---------------------------------------------------------
    # Remaining LaTeX command:
    #
    # \LayerNorm -> LayerNorm
    # ---------------------------------------------------------
    value = re.sub(
        r"\\([A-Za-z]+)",
        r"\1",
        value,
    )

    # ---------------------------------------------------------
    # Clean whitespace
    # ---------------------------------------------------------
    value = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        value,
    )

    value = re.sub(
        r"[ \t]+",
        " ",
        value,
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    value = "\n".join(
        line.strip()
        for line in value.splitlines()
        if line.strip()
    )

    return value.strip()