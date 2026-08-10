#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then cp .env.example .env; fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
cd frontend
npm install
cat <<MSG

Setup complete.
1) Edit ../.env and set GROQ_API_KEY.
2) Add English/Hindi documents to ../backend/data/docs/.
3) Validate: source ../.venv/bin/activate && python ../backend/scripts/validate_docs.py --path ../backend/data/docs
4) Ingest:   source ../.venv/bin/activate && python ../backend/scripts/ingest_docs.py --path ../backend/data/docs --reset
5) Backend:  source ../.venv/bin/activate && uvicorn app.main:app --app-dir ../backend --host 0.0.0.0 --port 8000 --reload
6) Frontend: npm run dev
MSG
