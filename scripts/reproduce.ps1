$ErrorActionPreference = 'Stop'
$interpreter = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $interpreter (Join-Path $PSScriptRoot 'reproduce.py') @args
exit $LASTEXITCODE
