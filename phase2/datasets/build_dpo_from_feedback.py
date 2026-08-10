import argparse, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True); a=p.parse_args()
src, dst = Path(a.input), Path(a.output); dst.parent.mkdir(parents=True, exist_ok=True)
written=0
with src.open(encoding='utf-8') as f, dst.open('w', encoding='utf-8') as out:
    for line in f:
        if not line.strip(): continue
        row=json.loads(line)
        corrected=(row.get('corrected_answer') or '').strip()
        rejected=(row.get('model_answer') or '').strip()
        if int(row.get('rating',0)) >= 0 or not corrected or not rejected or corrected == rejected:
            continue
        out.write(json.dumps({
            'prompt': row.get('query',''),
            'chosen': corrected,
            'rejected': rejected,
        }, ensure_ascii=False)+'\n')
        written += 1
print({'written': written, 'output': str(dst)})
