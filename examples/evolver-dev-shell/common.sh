#!/usr/bin/env bash

atrex_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
atrex_runtime_root="$(cd -- "${atrex_example_dir}/../.." && pwd)"
atrex_state_dir="${ATREX_EVOLVER_DEV_SHELL_STATE_DIR:-${atrex_runtime_root}/workspaces/evolver-dev-shell-example}"
atrex_env_file="${ATREX_EVOLVER_DEV_SHELL_ENV_FILE:-${atrex_state_dir}/runtime.env}"
atrex_config_file="${ATREX_EVOLVER_DEV_SHELL_CONFIG:-${atrex_state_dir}/runtime.json}"
atrex_runtime_template="${atrex_example_dir}/runtime.json"
atrex_campaign_file="${ATREX_CAMPAIGN:-${atrex_state_dir}/campaign.json}"
atrex_campaign_template="${atrex_example_dir}/campaign.json"
atrex_bootstrap_result_file="${atrex_state_dir}/unused-bootstrap-result.json"

# The temporary shell synthesizes one Active-only agent-v0 snapshot.
export ATREX_CHALLENGER_COUNT="${ATREX_CHALLENGER_COUNT:-0}"
export ATREX_CHALLENGER_START_EPOCH="${ATREX_CHALLENGER_START_EPOCH:-1}"
export ATREX_TRAJECTORIES_PER_BRANCH="${ATREX_TRAJECTORIES_PER_BRANCH:-1}"
export ATREX_ATTEMPTS_PER_TRAJECTORY="${ATREX_ATTEMPTS_PER_TRAJECTORY:-1}"

# shellcheck source=../shared/runtime-common.sh
source "${atrex_runtime_root}/examples/shared/runtime-common.sh"
