param([string]$Sam3Repo)

$ErrorActionPreference = 'Stop'
if (-not $Sam3Repo) {
    $Sam3Repo = Join-Path (Split-Path -Parent $PSScriptRoot) 'sam3'
}
# AppData can be redirected when setup runs inside a packaged desktop app.
# Use an interpreter location that ordinary PowerShell can also access.
if (-not $env:UV_PYTHON_INSTALL_DIR) {
    $env:UV_PYTHON_INSTALL_DIR = [Environment]::GetEnvironmentVariable('UV_PYTHON_INSTALL_DIR', 'User')
    if (-not $env:UV_PYTHON_INSTALL_DIR) {
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $env:USERPROFILE '.local\share\uv\python'
    }
}
Push-Location $PSScriptRoot
try {
    foreach ($environment in @('.venv','.venv-export')) {
        if (-not (Test-Path "$environment/Scripts/python.exe")) { uv venv --python 3.12 $environment }
        if ($LASTEXITCODE) { throw 'Environment creation failed.' }
        $lock = if ($environment -eq '.venv') { 'requirements-win-py312.lock.txt' } else { 'requirements-export-win-py312.lock.txt' }
        uv pip install --python "$environment/Scripts/python.exe" --index-strategy unsafe-best-match -r $lock
        if ($LASTEXITCODE) { throw "$environment dependencies failed." }
    }
    & .venv-export/Scripts/python.exe python/fetch_sam3_repo.py --sam3-repo $Sam3Repo
    if ($LASTEXITCODE) { throw 'Pinned SAM3 source checkout failed.' }
    uv pip install --python .venv-export/Scripts/python.exe --no-deps -e $Sam3Repo
    if ($LASTEXITCODE) { throw 'SAM3 exporter installation failed.' }
    & .venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE) { throw 'SAM3 tests failed.' }
} finally { Pop-Location }
