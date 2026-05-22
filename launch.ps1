$WorkspaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $WorkspaceDir

# Run setup if necessary, then launch
& "$WorkspaceDir\setup.ps1" -Run
