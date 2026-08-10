import os
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def read_dotenv_value(name: str) -> str:
    if not ENV_FILE.exists():
        return ""

    for raw_line in ENV_FILE.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split("=", 1)

        if key.strip() == name:
            return (
                value.strip()
                .strip('"')
                .strip("'")
            )

    return ""


api_key = (
    os.getenv("GROQ_API_KEY", "").strip()
    or read_dotenv_value("GROQ_API_KEY")
)

base_url = (
    read_dotenv_value("GROQ_BASE_URL")
    or "https://api.groq.com/openai/v1"
).rstrip("/")


if not api_key or api_key == "replace_me":
    print(
        "ERROR: GROQ_API_KEY is missing.\n"
        "Set it in the root .env file."
    )
    sys.exit(2)


url = f"{base_url}/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
    "User-Agent": "voice-rag-en-hi/1.0",
}


print("Checking Groq connectivity...")
print(f"Endpoint: {url}")
print("API key: configured (hidden)")


try:
    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
    ) as client:

        response = client.get(
            url,
            headers=headers,
        )

except httpx.RequestError as exc:
    print()
    print("ERROR: Could not connect to Groq.")
    print(exc)
    sys.exit(1)


if response.status_code != 200:
    print()
    print(
        f"ERROR: Groq returned HTTP "
        f"{response.status_code}"
    )

    # Never print Authorization/header/key.
    body = response.text[:1000]
    print(body)

    if "1010" in body:
        print()
        print(
            "Cloudflare error 1010 detected.\n"
            "The request is being blocked before normal "
            "Groq API authentication."
        )

    sys.exit(1)


payload = response.json()

model_ids = sorted(
    model.get("id", "")
    for model in payload.get("data", [])
)


print()
print("Groq API connection successful.")
print(f"Models visible: {len(model_ids)}")
print()


required_models = [
    "whisper-large-v3-turbo",
    "qwen/qwen3.6-27b",
]


for model in required_models:
    status = (
        "FOUND"
        if model in model_ids
        else "NOT FOUND"
    )

    print(f"{status:10} {model}")


print()
print("Groq API check complete.")