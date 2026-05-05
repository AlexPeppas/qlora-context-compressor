#!/usr/bin/env bash
# pod_bootstrap.sh — idempotent setup for a RunPod pod with the compressor_nw_vol
# network volume attached at /workspace.
#
# This script makes pod recreation fast: the first run installs everything into
# /workspace/.venv (which lives on the network volume, so survives pod death),
# subsequent runs just activate the venv. All HF / pip / torch caches also live
# on /workspace so wheels and model snapshots are downloaded only once, ever.
#
# Usage:
#   On a fresh pod, source this once:   `source /workspace/pod_bootstrap.sh`
#   Or wire it into ~/.bashrc so every new shell auto-activates:
#     `echo 'source /workspace/pod_bootstrap.sh' >> ~/.bashrc`
#
# Required env vars (set in RunPod's Pod Template > Environment Variables):
#   HF_TOKEN   — Hugging Face access token with Qwen gated-repo read scope
#                (write it once into a RunPod Secret, reference here)

# Do NOT use `set -e` — this script is meant to be SOURCED, not run.
# Any failure should not kill the user's interactive shell.

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/.venv"
PIP_CACHE="$WORKSPACE/.pip-cache"
COMPRESSOR_DIR="$WORKSPACE/compressor"

# ---------------------------------------------------------------------------
# 1. Persistent caches — point everything cacheable at the network volume
#
# Note: HF_HOME defaults to /workspace/.cache/huggingface to align with the
# location that earlier pod sessions used (and where the ~15 GB Qwen snapshot
# already lives). Forking it to a different path forces a re-download and
# nearly busted the network volume quota — see CLAUDE.md section 4c gotchas.
# ---------------------------------------------------------------------------
export HF_HOME="$WORKSPACE/.cache/huggingface"
export PIP_CACHE_DIR="$PIP_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$PIP_CACHE"

# ---------------------------------------------------------------------------
# 2. Venv: create on first run, activate on every run
# ---------------------------------------------------------------------------
# Sentinel marks a *completed* install. Checking just for the venv directory
# is unsafe: `python3 -m venv` creates the dir before pip ever runs, so any
# interruption (pod kill, OOM, network drop during the 2GB torch download)
# leaves a half-baked venv that the next bootstrap would happily activate
# and skip reinstall on. The sentinel is only written after the final pip
# install succeeds, so a partial state always re-triggers a full rebuild.
VENV_READY="$VENV/.bootstrap_complete"

if [ ! -f "$VENV_READY" ]; then
    if [ -d "$VENV" ]; then
        echo "[bootstrap] Detected incomplete venv at $VENV — wiping and rebuilding"
        rm -rf "$VENV"
    fi
    echo "[bootstrap] Creating venv at $VENV"
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    pip install --upgrade pip wheel || return 1 2>/dev/null || exit 1

    echo "[bootstrap] Installing pinned training stack (one-time, ~5 min)"
    # Order matters: torch FIRST with --no-deps to avoid being yanked back to
    # the latest +cu13 wheel by transitive deps from later installs.
    pip install --no-deps \
        torch==2.6.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
        || return 1 2>/dev/null || exit 1
    pip install \
        bitsandbytes==0.45.2 \
        transformers==4.45.2 \
        peft==0.14.0 \
        trl==0.11.4 \
        accelerate==1.0.1 \
        datasets==3.0.2 \
        hf_transfer \
        sentencepiece \
        python-dotenv \
        anthropic \
        sumy \
        nltk \
        || return 1 2>/dev/null || exit 1

    touch "$VENV_READY"
    echo "[bootstrap] venv install complete."
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# ---------------------------------------------------------------------------
# 3. Hugging Face auth — write token into HF cache if env var is set
#    (RunPod Secret -> Pod Template Env Var -> here)
# ---------------------------------------------------------------------------
if [ -n "$HF_TOKEN" ] && [ ! -f "$HF_HOME/token" ]; then
    mkdir -p "$HF_HOME"
    echo "$HF_TOKEN" > "$HF_HOME/token"
    echo "[bootstrap] HF token persisted to $HF_HOME/token"
fi

# ---------------------------------------------------------------------------
# 4. Convenience: jump to project dir and report status
# ---------------------------------------------------------------------------
if [ -d "$COMPRESSOR_DIR" ]; then
    cd "$COMPRESSOR_DIR" || cd "$WORKSPACE"
fi

echo "[bootstrap] env ready:"
echo "             python: $(python --version 2>&1)"
echo "             torch:  $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")' 2>/dev/null || echo '(torch not yet importable)')"
echo "             cwd:    $(pwd)"
