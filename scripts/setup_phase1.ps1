$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r backend\requirements.txt

Push-Location frontend
npm install
Pop-Location

Write-Host ""
Write-Host "Setup complete."
Write-Host "1) Edit .env and set GROQ_API_KEY."
Write-Host "2) Put English/Hindi documents in backend\data\docs\."
Write-Host "3) Validate: .\.venv\Scripts\python.exe backend\scripts\validate_docs.py --path backend\data\docs"
Write-Host "4) Ingest:   .\.venv\Scripts\python.exe backend\scripts\ingest_docs.py --path backend\data\docs --reset"
Write-Host "5) Backend:  .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload"
Write-Host "6) Frontend: cd frontend; npm run dev"
