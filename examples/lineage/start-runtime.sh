#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

atrex_example_require_remote_agate
atrex_example_require_agent_backend optimizer
atrex_example_load_local_secrets
atrex_example_prepare_inputs

if ! command -v "${atrex_runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${atrex_runtime_cli}" >&2
  exit 69
fi
exec "${atrex_runtime_cli}" serve --config "${atrex_config_file}"
