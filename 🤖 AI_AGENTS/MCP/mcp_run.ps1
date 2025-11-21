param(
  [Parameter(Mandatory=$true)] [string] $MetaEditor,
  [Parameter(Mandatory=$true)] [string] $Terminal,
  [Parameter(Mandatory=$true)] [string] $Spec,
  [string] $Out = "MCP/out"
)

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
  Write-Error "Python not found in PATH. Install Python 3.x first."
  exit 1
}

Write-Host "[MCP] Running backtest: $Spec" -ForegroundColor Cyan
python "$(Join-Path $PSScriptRoot 'mcp_backtest_runner.py')" --metaeditor "$MetaEditor" --terminal "$Terminal" --spec "$Spec" --out "$Out"

