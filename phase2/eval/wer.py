import argparse, json


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def tokens(text):
    return text.strip().lower().split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    errors = total = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ref, hyp = tokens(row["reference"]), tokens(row["hypothesis"])
            errors += edit_distance(ref, hyp)
            total += len(ref)
    print(f"WER: {errors / total if total else 0:.6f} ({errors}/{total})")


if __name__ == "__main__":
    main()
