import argparse
import json
from pathlib import Path

SYSTEM = (
    "You are an English/Hindi enterprise document assistant. "
    "Use verified knowledge for enterprise facts, never invent unsupported facts, "
    "answer in the requested/user language, and keep common technical terms in English when natural."
)

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

src, dst = Path(a.input), Path(a.output)
dst.parent.mkdir(parents=True, exist_ok=True)
written = 0
with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        rating = int(row.get("rating", 0))
        corrected = (row.get("corrected_answer") or "").strip()
        model_answer = (row.get("model_answer") or "").strip()
        if rating > 0 and model_answer:
            target = model_answer
        elif rating < 0 and corrected:
            target = corrected
        else:
            continue
        language = row.get("language", "en")
        item = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Requested language: {language}\n\n{row['query']}"},
                {"role": "assistant", "content": target},
            ]
        }
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1
print({"written": written, "output": str(dst)})
