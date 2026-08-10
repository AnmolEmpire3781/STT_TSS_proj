.PHONY: backend frontend ingest validate compile zip
backend:
	uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
frontend:
	cd frontend && npm run dev
validate:
	python backend/scripts/validate_docs.py --path backend/data/docs
ingest:
	python backend/scripts/ingest_docs.py --path backend/data/docs --reset
compile:
	python -m compileall backend phase2 scripts
zip:
	cd .. && zip -r voice-rag-en-hi.zip voice-rag-en-hi -x '*/node_modules/*' '*/.venv*/*' '*/__pycache__/*' '*.pyc'
