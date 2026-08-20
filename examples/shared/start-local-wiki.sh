#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="$(cd -- "${script_dir}/../.." && pwd)"
reference_root="${runtime_root}/workspaces/local-wiki/corpus/gpu-wiki"
python_executable="${ATREX_PYTHON:-python3}"

if ! command -v "${python_executable}" >/dev/null 2>&1; then
  echo "missing development Python in PATH: ${python_executable}" >&2
  echo "activate a platform-local environment with the Runtime dependencies first" >&2
  exit 2
fi

if ! "${python_executable}" -c 'import anyio, pydantic, uvicorn' >/dev/null 2>&1; then
  echo "Local Wiki dependencies are unavailable in: $(command -v "${python_executable}")" >&2
  echo "Create and activate a Linux-local virtual environment, then install:" >&2
  echo "  python -m pip install -e '.[dev]' -e './workspaces/local-wiki[dev]'" >&2
  echo "Or set ATREX_PYTHON to an interpreter containing these dependencies." >&2
  exit 69
fi

if [[ ! -d "${reference_root}" ]]; then
  echo "missing vendored GPU Wiki corpus: ${reference_root}" >&2
  echo "the Runtime checkout is incomplete; restore workspaces/local-wiki/corpus" >&2
  exit 2
fi

exec env \
  PYTHONPATH="${runtime_root}/workspaces/local-wiki/src" \
  "${python_executable}" -m atrex_local_wiki serve \
  --config "${runtime_root}/workspaces/local-wiki/configs/local.example.json"
