#!/usr/bin/env bash
# Source this, do not execute it:  source scripts/env.sh
#
# Points every cache at a volume with room. On the dev laptop C: has ~10 GB
# free and D: has ~147 GB, and both the HF model cache and the activation
# cache default to C: if left alone.

QP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QUANTPROBE_ROOT="${QP_ROOT}"
export QUANTPROBE_CACHE="${QUANTPROBE_CACHE:-D:/quantprobe_cache}"
export HF_HOME="${HF_HOME:-D:/quantprobe_cache/hf}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Deterministic cuBLAS reductions. Must be set before the first CUDA call,
# so it is exported here as well as in src/seeding.py.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

mkdir -p "${QUANTPROBE_CACHE}" "${HF_HOME}" 2>/dev/null

echo "QUANTPROBE_ROOT  = ${QUANTPROBE_ROOT}"
echo "QUANTPROBE_CACHE = ${QUANTPROBE_CACHE}"
echo "HF_HOME          = ${HF_HOME}"
if [ -n "${HF_TOKEN}" ]; then
  echo "HF_TOKEN         = set (${#HF_TOKEN} chars)"
else
  echo "HF_TOKEN         = NOT SET - gated models (Llama-3.2) will fail"
fi
