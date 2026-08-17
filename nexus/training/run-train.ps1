<#
.SYNOPSIS
    Launches a NEXUS fine-tune so that PowerShell cannot kill it and so the
    output survives the terminal.

.DESCRIPTION
    Forces $ErrorActionPreference = 'Continue' so a native stderr warning
    cannot end the run, tees everything to a timestamped log, and refuses to
    start under the repo venv's CPU-only torch.

.EXAMPLE
    .\nexus\training\run-train.ps1 -Config nexus\training\configs\qwen3b-local.yaml -Dataset nexus\training\data\v1.jsonl
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Dataset,
    [switch]$Resume,
    [switch]$DryRun,
    [string]$Python = "C:\Users\sajjadlh8\AppData\Local\Programs\Python\Python311\python.exe"
)

$ErrorActionPreference = 'Continue'

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

foreach ($pair in @(@{n = "Config"; v = $Config }, @{n = "Dataset"; v = $Dataset })) {
    if (-not (Test-Path $pair.v)) {
        Write-Host "FATAL: -$($pair.n) not found: $($pair.v)" -ForegroundColor Red
        exit 2
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "FATAL: Python interpreter not found: $Python" -ForegroundColor Red
    Write-Host "       Pass -Python <path-to-cuda-python.exe> if it has moved." -ForegroundColor Red
    exit 2
}

$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$configName = [System.IO.Path]::GetFileNameWithoutExtension($Config)
$logPath = Join-Path $logDir "train-$configName-$stamp.log"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " NEXUS training run" -ForegroundColor Cyan
Write-Host "   config      : $Config"
Write-Host "   dataset     : $Dataset"
Write-Host "   interpreter : $Python"
Write-Host "   LOG         : $logPath" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$probe = & $Python -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.__version__); print(ok); print(torch.cuda.get_device_name(0) if ok else 'none')"
$probeLines = @($probe) -split "`r?`n" | Where-Object { $_ -ne "" }
if ($LASTEXITCODE -ne 0 -or $probeLines.Count -lt 3) {
    Write-Host "FATAL: could not import torch with $Python" -ForegroundColor Red
    $probe | Out-String | Write-Host
    exit 3
}
if ($probeLines[1].Trim() -ne "True") {
    Write-Host "FATAL: torch.cuda.is_available() is False under $Python" -ForegroundColor Red
    Write-Host "       torch reports: $($probeLines[0])" -ForegroundColor Red
    Write-Host "       This is almost certainly the CPU-only torch in the repo venv." -ForegroundColor Red
    Write-Host "       Training on CPU would take days; refusing to start." -ForegroundColor Red
    exit 3
}
Write-Host "torch $($probeLines[0]) | CUDA OK | $($probeLines[2])" -ForegroundColor Green
Write-Host ""

$trainArgs = @("-u", "-m", "nexus.training.train", "--dataset", $Dataset, "--config", $Config)
if ($Resume) { $trainArgs += "--resume" }
if ($DryRun) { $trainArgs += "--dry-run" }

"=== NEXUS training run $stamp ===" | Out-File -FilePath $logPath -Encoding utf8
"config=$Config dataset=$Dataset python=$Python" | Out-File -FilePath $logPath -Encoding utf8 -Append
"torch=$($probeLines[0]) gpu=$($probeLines[2])" | Out-File -FilePath $logPath -Encoding utf8 -Append
"" | Out-File -FilePath $logPath -Encoding utf8 -Append

$started = Get-Date

& $Python $trainArgs 2>&1 | ForEach-Object { $_ | Out-String } | Tee-Object -FilePath $logPath -Append
$exit = $LASTEXITCODE

$elapsed = (Get-Date) - $started
$summary = "exit_code=$exit wall_time=$([math]::Round($elapsed.TotalMinutes,2))min"
$summary | Out-File -FilePath $logPath -Encoding utf8 -Append

Write-Host ""
if ($exit -eq 0) {
    Write-Host "Run finished OK. $summary" -ForegroundColor Green
}
else {
    Write-Host "Run FAILED. $summary" -ForegroundColor Red
    if ($exit -eq -1073741819) {
        Write-Host "exit -1073741819 = 0xC0000005 ACCESS VIOLATION (native crash, no traceback)." -ForegroundColor Red
    }
}
Write-Host "Log: $logPath" -ForegroundColor Yellow
exit $exit
