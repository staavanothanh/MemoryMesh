$WorkspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Isolate opencode sessions per workspace
$env:OPENCODE_DATA_DIR = "$WorkspaceDir\.opencode\data"

# Navigate to workspace root
Set-Location $WorkspaceDir

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " MemoryMesh Sandbox Activated" -ForegroundColor Green
Write-Host " Workspace: $WorkspaceDir" -ForegroundColor Yellow
Write-Host " Data: $env:OPENCODE_DATA_DIR" -ForegroundColor DarkGray
Write-Host "=====================================================" -ForegroundColor Cyan

opencode
