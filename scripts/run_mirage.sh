#!/usr/bin/env bash
# Convenience launcher for MIRAGE runs.
#
# This wrapper ensures the repository root is on PYTHONPATH and that the
# vLLM autostart log directory exists before delegating to main_mirage.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${REPO_ROOT}/logs/vllm"

python "${REPO_ROOT}/main_mirage.py" "$@"
