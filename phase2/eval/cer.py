import argparse
import json
import re


def normalize(s):
    return re.sub(r"\s+", "", s.strip().lower())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "--file", dest="input", required=True, help="JSONL with reference and hypothesis")
    a = p.parse_args()
    errors = chars = 0
    with open(a.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = normalize(row["reference"])
            hyp = normalize(row["hypothesis"])
            errors += edit_distance(ref, hyp)
            chars += len(ref)
    print(f"CER: {errors / max(chars, 1):.6f} ({errors}/{chars})")


if __name__ == "__main__":
    main()
