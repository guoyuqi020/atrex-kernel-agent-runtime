#!/usr/bin/env bash

atrex_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
atrex_runtime_root="$(cd -- "${atrex_example_dir}/../.." && pwd)"
atrex_state_dir="${ATREX_LINEAGE_STATE_DIR:-${atrex_runtime_root}/workspaces/lineage-example}"
atrex_env_file="${ATREX_LINEAGE_ENV_FILE:-${atrex_state_dir}/runtime.env}"
atrex_config_file="${ATREX_LINEAGE_CONFIG:-${atrex_state_dir}/runtime.json}"
atrex_runtime_template="${atrex_example_dir}/runtime.json"
atrex_campaign_file="${ATREX_CAMPAIGN:-${atrex_state_dir}/campaign.json}"
atrex_campaign_template="${atrex_example_dir}/campaign.json"
atrex_bootstrap_result_file="${ATREX_LINEAGE_BOOTSTRAP_RESULT:-${atrex_state_dir}/bootstrap-result.json}"

# Runtime supplies only a resource envelope. The versioned Agent Workflow spends it.
export ATREX_MAX_CHALLENGERS="${ATREX_MAX_CHALLENGERS:-0}"
export ATREX_OPTIMIZER_ATTEMPT_BUDGET="${ATREX_OPTIMIZER_ATTEMPT_BUDGET:-3}"

# shellcheck source=../shared/runtime-common.sh
source "${atrex_runtime_root}/examples/shared/runtime-common.sh"

lineage_epoch_result_file="${ATREX_LINEAGE_EPOCH_RESULT:-${atrex_state_dir}/epoch-1-result.json}"

lineage_require_result() {
  if [[ ! -f "${lineage_epoch_result_file}" ]]; then
    echo "Epoch result not found: ${lineage_epoch_result_file}" >&2
    echo "Run examples/lineage/run-epoch.sh first." >&2
    return 66
  fi
}
