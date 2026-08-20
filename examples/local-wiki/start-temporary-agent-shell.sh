#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_load_env

config_path="$(atrex_demo_config_path)"
python_executable="${atrex_demo_python}"
if [[ ! -f "${config_path}" ]]; then
  echo "Runtime config not found: ${config_path}" >&2
  exit 66
fi
if ! command -v "${python_executable}" >/dev/null 2>&1; then
  echo "Runtime Python not found in PATH: ${python_executable}" >&2
  exit 69
fi

exec "${python_executable}" "${script_dir}/temporary_wiki_shell.py" \
  --config "${config_path}" "$@"
