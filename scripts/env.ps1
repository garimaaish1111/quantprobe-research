# Dot-source this, do not run it:  . .\scripts\env.ps1
#
# Points every cache at a volume with room. On the dev laptop C: has ~10 GB
# free and D: has ~147 GB, and both the HF model cache and the activation
# cache default to C: if left alone.

$QpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$env:QUANTPROBE_ROOT = $QpRoot
if (-not $env:QUANTPROBE_CACHE) { $env:QUANTPROBE_CACHE = "D:\quantprobe_cache" }
if (-not $env:HF_HOME)          { $env:HF_HOME          = "D:\quantprobe_cache\hf" }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Deterministic cuBLAS reductions. Must be set before the first CUDA call,
# so it is exported here as well as in src/seeding.py.
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

New-Item -ItemType Directory -Force -Path $env:QUANTPROBE_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

"QUANTPROBE_ROOT  = $($env:QUANTPROBE_ROOT)"
"QUANTPROBE_CACHE = $($env:QUANTPROBE_CACHE)"
"HF_HOME          = $($env:HF_HOME)"
if ($env:HF_TOKEN) {
  "HF_TOKEN         = set ($($env:HF_TOKEN.Length) chars)"
} else {
  "HF_TOKEN         = NOT SET - gated models (Llama-3.2) will fail"
}
