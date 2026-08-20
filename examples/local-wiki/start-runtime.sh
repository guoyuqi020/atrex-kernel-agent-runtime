#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_load_env

config_path="$(atrex_demo_config_path)"
runtime_cli="${atrex_demo_runtime_cli}"
if [[ ! -f "${config_path}" ]]; then
  echo "Runtime config not found: ${config_path}" >&2
  exit 66
fi
if ! command -v "${runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${runtime_cli}" >&2
  exit 69
fi

exec "${runtime_cli}" serve --config "${config_path}"
