# Thin wrapper around docker for the GPU workflow. Usage: ./run.ps1 <target> [args...]
#   build      build the image
#   smoke      run the Blackwell smoke-test gate
#   test       run the pytest suite in-container
#   shell      interactive shell in-container
#   cli        run the package CLI in-container (pass-through args)
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Image = "cuda-motion-flow:dev"
$Root = $PSScriptRoot

$Mounts = @(
    "-v", "${Root}:/workspace",
    "-v", "${Root}/data:/workspace/data",
    "-v", "${Root}/weights:/workspace/weights",
    "-v", "${Root}/outputs:/workspace/outputs"
)
$RunArgs = @("run", "--rm", "--gpus", "all") + $Mounts + @("-w", "/workspace", $Image)

switch ($Target) {
    "build" { docker build -t $Image $Root }
    "smoke" { docker @RunArgs python3 scripts/smoke_test.py }
    "test"  { docker @RunArgs python3 -m pytest @Rest }
    "shell" { docker @($RunArgs[0..($RunArgs.Length - 2)] + @("-it", $Image, "bash")) }
    "cli"   { docker @RunArgs python3 -m cuda_motion_flow.cli @Rest }
    default { Write-Error "unknown target '$Target' (build|smoke|test|shell|cli)" }
}
