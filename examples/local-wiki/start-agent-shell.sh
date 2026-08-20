#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_load_env

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [lineage-id] [zsh|bash]" >&2
  exit 64
fi

lineage_id=""
shell_name="zsh"
if [[ $# -eq 1 && ("$1" == "zsh" || "$1" == "bash") ]]; then
  shell_name="$1"
elif [[ $# -ge 1 ]]; then
  lineage_id="$1"
  shell_name="${2:-zsh}"
fi
if [[ -z "${lineage_id}" ]]; then
  lineage_id="$(atrex_demo_last_lineage_id)"
fi
runtime_cli="${atrex_demo_runtime_cli}"
config_path="$(atrex_demo_config_path)"

if [[ ! -f "${config_path}" ]]; then
  echo "Runtime config not found: ${config_path}" >&2
  echo "Set ATREX_RUNTIME_CONFIG to an existing Runtime JSON config." >&2
  exit 66
fi
if ! command -v "${runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${runtime_cli}" >&2
  exit 69
fi

exec "${runtime_cli}" dev-shell \
  --config "${config_path}" \
  --lineage "${lineage_id}" \
  --shell "${shell_name}"
