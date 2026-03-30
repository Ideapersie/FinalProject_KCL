#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# setup_environment.sh — Sprint 1 environment setup
#
# Usage:
#   bash scripts/setup_environment.sh [--low-tier] [--skip-model]
#
# Options:
#   --low-tier    Install Llama 3.2-1B (Q4) instead of 3B model
#   --skip-model  Skip GGUF model download (useful on CI / no disk space)
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────
LOW_TIER=false
SKIP_MODEL=false
for arg in "$@"; do
  case $arg in
    --low-tier)   LOW_TIER=true  ;;
    --skip-model) SKIP_MODEL=true ;;
  esac
done

# ── Detect project root ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# ── Python version check ──────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
  echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
  exit 1
fi
echo "Python $PYTHON_VERSION — OK"

# ── Create virtual environment ────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Activated .venv"

# ── Upgrade pip ───────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel --quiet

# ── Install llama-cpp-python (CPU-only) ───────────────────────────
# Check if OpenBLAS is available for faster CPU inference
if command -v dpkg &>/dev/null && dpkg -l libopenblas-dev &>/dev/null 2>&1; then
  echo "OpenBLAS detected — building llama-cpp-python with BLAS support"
  CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install llama-cpp-python --quiet
else
  echo "OpenBLAS not found — installing llama-cpp-python without BLAS (slower)"
  echo "  To install OpenBLAS: sudo apt-get install libopenblas-dev"
  pip install llama-cpp-python --quiet
fi

# ── Install remaining dependencies ───────────────────────────────
pip install -r requirements.txt --quiet
echo "Dependencies installed"

# ── Install package in editable mode ─────────────────────────────
pip install -e ".[dev]" --quiet
echo "Package installed in editable mode"

# ── Create required directories ───────────────────────────────────
mkdir -p \
  data/raw/mirage \
  data/raw/ragcare_qa \
  data/raw/acute_care_raw \
  data/processed \
  data/corpora/statpearls \
  data/corpora/textbooks \
  data/corpora/bnf_chunks \
  data/corpora/who_mhgap_chunks \
  indexes \
  models \
  results/raw_logs \
  results/aggregated \
  results/figures
echo "Directories created"

# ── Download GGUF model ───────────────────────────────────────────
if [ "$SKIP_MODEL" = false ]; then
  if [ "$LOW_TIER" = true ]; then
    MODEL_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    MODEL_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    MODEL_DEST="models/llama-3.2-1b-q4_k_m.gguf"
  else
    MODEL_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
    MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    MODEL_DEST="models/llama-3.2-3b-q4_k_m.gguf"
  fi

  if [ -f "$MODEL_DEST" ]; then
    echo "Model already exists: $MODEL_DEST — skipping download"
  else
    echo "Downloading $MODEL_FILE from HuggingFace ..."
    echo "  Repo: $MODEL_REPO"
    echo "  Destination: $MODEL_DEST"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='$MODEL_REPO',
    filename='$MODEL_FILE',
    local_dir='models/',
)
print(f'Downloaded to: {path}')
"
    echo "Model downloaded: $MODEL_DEST"
  fi
else
  echo "Skipping model download (--skip-model)"
fi

# ── Verify installation ───────────────────────────────────────────
echo ""
echo "─── Verification ────────────────────────────────────────────"
python3 -c "import medrag_adaptive; print('medrag_adaptive package: OK')"
python3 -c "import yaml; print('pyyaml: OK')"
python3 -c "import pydantic; print('pydantic: OK')"
python3 -c "
try:
    import llama_cpp
    print('llama-cpp-python: OK')
except ImportError:
    print('llama-cpp-python: NOT INSTALLED (install manually)')
"
python3 -c "
try:
    import faiss
    print('faiss-cpu: OK')
except ImportError:
    print('faiss-cpu: NOT INSTALLED')
"
python3 -c "
try:
    import codecarbon
    print('codecarbon: OK')
except ImportError:
    print('codecarbon: NOT INSTALLED')
"

# ── Run unit tests ────────────────────────────────────────────────
echo ""
echo "─── Running unit tests ──────────────────────────────────────"
python3 -m pytest tests/unit/ -v --tb=short
echo ""
echo "─────────────────────────────────────────────────────────────"
echo "Setup complete!"
echo ""
echo "Sprint 1 milestone — test closed-book inference:"
if [ -f "models/llama-3.2-3b-q4_k_m.gguf" ]; then
  echo "  python -m medrag_adaptive.models.llama_backend \\"
  echo "      --model models/llama-3.2-3b-q4_k_m.gguf \\"
  echo "      --prompt 'What is the mechanism of action of metformin?'"
elif [ -f "models/llama-3.2-1b-q4_k_m.gguf" ]; then
  echo "  python -m medrag_adaptive.models.llama_backend \\"
  echo "      --model models/llama-3.2-1b-q4_k_m.gguf \\"
  echo "      --prompt 'What is the mechanism of action of metformin?'"
else
  echo "  (Download a model first, then run the command above)"
fi
