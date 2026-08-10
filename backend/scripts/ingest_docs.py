import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag.ingest import ingest_path

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

stats = ingest_path(Path(args.path).expanduser().resolve(), reset=args.reset)
print(stats)
