#!/usr/bin/env bash
# Configure CUDA/cuDNN environment for this venv session.
# Usage: source scripts/gpu_env.sh
set -euo pipefail

# Ensure we're in the repo root (script can be sourced from anywhere)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_BIN="$REPO_DIR/.venv/bin"
PY="$VENV_BIN/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv Python not found at $PY" >&2
  return 1 2>/dev/null || exit 1
fi

# Find cuDNN wheel lib dir inside this venv
CUDNN_LIB_DIR="$($PY - <<'PY'
import importlib.util, os
spec = importlib.util.find_spec('nvidia.cudnn')
print(os.path.join(spec.submodule_search_locations[0], 'lib') if spec and spec.submodule_search_locations else '')
PY
)"

if [[ -z "$CUDNN_LIB_DIR" || ! -d "$CUDNN_LIB_DIR" ]]; then
  echo "ERROR: Could not locate cuDNN lib directory in venv. Install with:"
  echo "  $VENV_BIN/pip install nvidia-cudnn-cu12==9.1.0.70"
  return 1 2>/dev/null || exit 1
fi

# Prepend cuDNN libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH prepared with cuDNN: $CUDNN_LIB_DIR"

# Optional: show CUDA visibility
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
fi
