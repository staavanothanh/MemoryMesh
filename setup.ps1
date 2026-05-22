param(
    [switch]$Run
)

$WorkspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $WorkspaceDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MemoryMesh Setup (Windows)" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan

# --- Create .env if missing ---
if (-not (Test-Path ".env")) {
    Write-Host "[1/4] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "      -> Edit .env to set your LLM endpoint and model." -ForegroundColor Gray
} else {
    Write-Host "[1/4] .env already exists, skipping." -ForegroundColor Gray
}

# --- Create virtual environment ---
if (-not (Test-Path ".venv")) {
    Write-Host "[2/4] Creating Python virtual environment..." -ForegroundColor Yellow
    & python -m venv .venv
} else {
    Write-Host "[2/4] Virtual environment already exists, skipping." -ForegroundColor Gray
}

# --- Activate & install ---
Write-Host "[3/4] Installing MemoryMesh..." -ForegroundColor Yellow
$Activate = Join-Path $WorkspaceDir ".venv" "Scripts" "Activate.ps1"
. $Activate
& pip install -e ".[test]"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "  Run:       python -m memorymesh"
Write-Host "  Test:      python -m pytest tests/ -v"
Write-Host ""
Write-Host "  Tip:       Restart your terminal after first setup." -ForegroundColor DarkGray
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " MemoryMesh Sandbox Activated" -ForegroundColor Green
Write-Host " Workspace: $WorkspaceDir" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan

if ($Run) {
    # Isolate opencode sessions per workspace
    $env:OPENCODE_DATA_DIR = "$WorkspaceDir\.opencode\data"
    if (Get-Command "opencode" -ErrorAction SilentlyContinue) {
        opencode
    } else {
        Write-Host "opencode not found in PATH; skipping launch." -ForegroundColor Red
    }
}
