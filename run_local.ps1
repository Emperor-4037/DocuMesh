# AI Writing Assistant - Local Dev Launcher
# Starts all Python backend services + React frontend WITHOUT Docker.
# Usage: .\run_local.ps1

$ROOT = $PSScriptRoot

# Shared environment for every service
$env:REDIS_HOST              = "localhost"
$env:REDIS_PORT              = "6379"
$env:QDRANT_HOST             = "localhost"
$env:QDRANT_PORT             = "6333"
$env:POSTGRES_SERVER         = "localhost"
$env:POSTGRES_PORT           = "5433"
$env:POSTGRES_USER           = "postgres"
$env:POSTGRES_PASSWORD       = "postgres"
$env:POSTGRES_DB             = "aiplatform"
$env:SECRET_KEY              = "super-secret-key-change-in-production"
$env:PYTHONPATH              = $ROOT

# Gateway: overrides for local service URLs
$env:PARAPHRASE_SERVICE_URL  = "http://localhost:8001"
$env:GRAMMAR_SERVICE_URL     = "http://localhost:8002"
$env:SIMPLIFY_SERVICE_URL    = "http://localhost:8003"
$env:TONE_SERVICE_URL        = "http://localhost:8004"
$env:SUMMARIZE_SERVICE_URL   = "http://localhost:8005"
$env:RAG_SERVICE_URL         = "http://localhost:8006"

function Start-LocalService {
    param(
        [string]$WinTitle,
        [string]$WorkDir,
        [string]$Cmd
    )
    $script = "cd `"$WorkDir`"; $Cmd"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $script -WindowStyle Normal
    Start-Sleep -Milliseconds 400
}

Write-Host "Starting AI Writing Assistant locally..." -ForegroundColor Cyan
Write-Host ""

# 1. Gateway on port 8000
Write-Host "  [1/8] Gateway          -> http://localhost:8000" -ForegroundColor Green
Start-LocalService -WinTitle "Gateway" -WorkDir $ROOT -Cmd "uvicorn gateway.app.main:app --host 0.0.0.0 --port 8000 --reload"

# 2. Paraphrase on port 8001
Write-Host "  [2/8] Paraphrase svc   -> http://localhost:8001" -ForegroundColor Green
Start-LocalService -WinTitle "Paraphrase" -WorkDir $ROOT -Cmd "uvicorn services.paraphrase_service.app.main:app --host 0.0.0.0 --port 8001 --reload"

# 3. Grammar on port 8002
Write-Host "  [3/8] Grammar svc      -> http://localhost:8002" -ForegroundColor Green
Start-LocalService -WinTitle "Grammar" -WorkDir $ROOT -Cmd "uvicorn services.grammar_service.app.main:app --host 0.0.0.0 --port 8002 --reload"

# 4. Simplify on port 8003
Write-Host "  [4/8] Simplify svc     -> http://localhost:8003" -ForegroundColor Green
Start-LocalService -WinTitle "Simplify" -WorkDir $ROOT -Cmd "uvicorn services.simplify_service.app.main:app --host 0.0.0.0 --port 8003 --reload"

# 5. Tone on port 8004
Write-Host "  [5/8] Tone svc         -> http://localhost:8004" -ForegroundColor Green
Start-LocalService -WinTitle "Tone" -WorkDir $ROOT -Cmd "uvicorn services.tone_service.app.main:app --host 0.0.0.0 --port 8004 --reload"

# 6. Summarize on port 8005
Write-Host "  [6/8] Summarize svc    -> http://localhost:8005" -ForegroundColor Green
Start-LocalService -WinTitle "Summarize" -WorkDir $ROOT -Cmd "uvicorn services.summarize_service.app.main:app --host 0.0.0.0 --port 8005 --reload"

# 7. RAG on port 8006
Write-Host "  [7/8] RAG svc          -> http://localhost:8006" -ForegroundColor Green
Start-LocalService -WinTitle "RAG" -WorkDir $ROOT -Cmd "uvicorn services.rag_service.app.main:app --host 0.0.0.0 --port 8006 --reload"

# 8. Frontend on port 5173
Write-Host "  [8/8] Frontend (Vite)  -> http://localhost:5173" -ForegroundColor Green
Start-LocalService -WinTitle "Frontend" -WorkDir "$ROOT\frontend" -Cmd "npm run dev"

Write-Host ""
Write-Host "All services launched in separate terminal windows." -ForegroundColor Cyan
Write-Host ""
Write-Host "  API Gateway:  http://localhost:8000"       -ForegroundColor Yellow
Write-Host "  API Docs:     http://localhost:8000/docs"  -ForegroundColor Yellow
Write-Host "  Frontend UI:  http://localhost:5173"       -ForegroundColor Yellow
Write-Host ""
Write-Host "NOTE: NLP services will take 1-3 min on first start" -ForegroundColor DarkYellow
Write-Host "      (downloading AI models from HuggingFace)."      -ForegroundColor DarkYellow
Write-Host ""
