# Starts the canguard dashboard (API + frontend) on Windows.
# Usage:  .\run.ps1            (default port 8000)
#         .\run.ps1 -Port 9000

param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found. Create it first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

$python = ".venv\Scripts\python.exe"

if (-not (Test-Path "data\processed\windows.parquet")) {
    Write-Host "Processed data missing - running parser + features (one-time)..."
    & $python -m src.data.parser
    & $python -m src.data.features
}
if (-not (Test-Path "models\lstm_ae.onnx")) {
    Write-Host "Models missing - training (one-time, a few minutes on CPU)..."
    & $python -m src.models.train
}
if (-not (Test-Path "knowledge_base\chroma")) {
    Write-Host "Knowledge base missing - building (one-time)..."
    & $python -m src.rag.build_kb
}

Write-Host ""
Write-Host "canguard -> http://127.0.0.1:$Port" -ForegroundColor Green
& $python -m uvicorn src.api.main:app --host 127.0.0.1 --port $Port
