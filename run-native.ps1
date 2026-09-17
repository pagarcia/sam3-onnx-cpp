param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'build/bin/Segment.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the native application first; see WORKSTATION.md.' }
$env:Path = (Join-Path $PSScriptRoot '.venv/Lib/site-packages/torch/lib') + ';' + $env:Path
if (-not $Arguments) { $Arguments = @('--help') }
Push-Location $PSScriptRoot
try { & $exe @Arguments; exit $LASTEXITCODE } finally { Pop-Location }
