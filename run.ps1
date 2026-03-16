# Thin wrapper around docker for the GPU workflow. Usage: ./run.ps1 <target> [args...]
#   image      build the docker image
#   ext        compile the cuda extension into src/ (needed before test/cli)
#   smoke      run the Blackwell smoke-test gate
#   test       build the extension then run pytest in-container
#   shell      interactive shell in-container
#   cli        run the package CLI in-container (pass-through args)
param(
    [Parameter(Mandatory = $true)][string]$Target,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
$Image = "gimbal:dev"
$Root = $PSScriptRoot

$Mounts = @(
    "-v", "${Root}:/workspace",
    "-v", "${Root}/data:/workspace/data",
    "-v", "${Root}/outputs:/workspace/outputs"
)
$RunArgs = @("run", "--rm", "--gpus", "all") + $Mounts + @("-w", "/workspace", $Image)
$EnsureExt = "python3 setup.py build_ext --inplace"

switch ($Target) {
    "image" { docker build -t $Image $Root }
    "ext"   { docker @RunArgs python3 setup.py build_ext --inplace }
    "smoke" { docker @RunArgs python3 scripts/smoke_test.py }
    "test"  { docker @RunArgs bash -lc "$EnsureExt && python3 -m pytest $($Rest -join ' ')" }
    "shell" { docker @($RunArgs[0..($RunArgs.Length - 2)] + @("-it", $Image, "bash")) }
    "cli"   { docker @RunArgs bash -lc "$EnsureExt && python3 -m gimbal.cli $($Rest -join ' ')" }
    default { Write-Error "unknown target '$Target' (image|ext|smoke|test|shell|cli)" }
}
