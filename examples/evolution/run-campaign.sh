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
atrex_example_require_agent_backend evolver
atrex_example_load_local_secrets
if [[ "${inputs_already_prepared}" == false ]]; then
  atrex_example_prepare_inputs
elif [[ ! -f "${atrex_config_file}" || ! -f "${atrex_campaign_file}" ]]; then
  echo "prepared Runtime config or Campaign definition is missing" >&2
  exit 66
fi

if ! command -v "${atrex_runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${atrex_runtime_cli}" >&2
  exit 69
fi
atrex_example_require_runtime_health
atrex_example_ensure_bootstrapped_campaign
selected_lineage_id="$(atrex_example_lineage_id)"

echo
echo "Running through Epoch 3: K=${ATREX_CHALLENGER_COUNT} from Epoch ${ATREX_CHALLENGER_START_EPOCH}, Y=${ATREX_TRAJECTORIES_PER_BRANCH}, X=${ATREX_ATTEMPTS_PER_TRAJECTORY}"
mkdir -p "$(dirname -- "${evolution_result_file}")"
temporary="${evolution_result_file}.tmp.$$"
trap 'rm -f "${temporary}"' EXIT
umask 077
"${atrex_runtime_cli}" run-campaign \
  --config "${atrex_config_file}" \
  --lineage "${selected_lineage_id}" \
  --target-epoch 3 | tee "${temporary}"
mv "${temporary}" "${evolution_result_file}"
trap - EXIT

echo
echo "Evolution result saved to: ${evolution_result_file}"
echo "Inspect it with: bash examples/evolution/inspect.sh"
