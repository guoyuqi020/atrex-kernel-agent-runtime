#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

inputs_already_prepared=false
if (( $# > 1 )); then
  echo "usage: $0 [--inputs-already-prepared]" >&2
  exit 64
fi
if (( $# == 1 )); then
  if [[ "$1" != "--inputs-already-prepared" ]]; then
    echo "usage: $0 [--inputs-already-prepared]" >&2
    exit 64
  fi
  inputs_already_prepared=true
fi

atrex_example_require_remote_agate
atrex_example_require_agent_backend optimizer
atrex_example_load_local_secrets
if [[ "${inputs_already_prepared}" == false ]]; then
  atrex_example_prepare_inputs
elif [[ ! -f "${atrex_config_file}" || ! -f "${atrex_campaign_file}" ]]; then
  echo "prepared Runtime config or Campaign definition is missing" >&2
  exit 66
fi

atrex_example_bootstrap_campaign
echo "Inspect it with: bash examples/bootstrap/inspect.sh"
