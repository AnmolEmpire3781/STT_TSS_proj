import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag.chunking import load_sections

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv"}


def main():
    parser = argparse.ArgumentParser(description="Check whether starter document parsers can extract useful text.")
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    root = Path(args.path).expanduser().resolve()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())

    readable = 0
    failed = 0
    for path in files:
        if path.name.startswith(".") or path.name == "README.md":
            print(f"SKIP  {path}  helper/hidden file")
            continue
        if path.suffix.lower() not in SUPPORTED:
            print(f"SKIP  {path}  unsupported extension")
            continue
        try:
            sections = load_sections(path)
            chars = sum(len(s.text) for s in sections)
            if chars < 30:
                failed += 1
                print(f"WARN  {path}  extracted only {chars} characters; scanned/image-only file is possible")
            else:
                readable += 1
                print(f"OK    {path}  sections={len(sections)} chars={chars}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {path}  {exc}")

    print(f"\nSummary: readable={readable}, warnings/failures={failed}")


if __name__ == "__main__":
    main()
