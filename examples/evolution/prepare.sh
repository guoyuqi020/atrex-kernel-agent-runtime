#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

atrex_example_require_remote_agate
atrex_example_require_agent_backend optimizer
atrex_example_require_agent_backend evolver
atrex_example_load_local_secrets
atrex_example_prepare_inputs

echo "Evolution schedule: Epoch 1 Active-only; Epochs 2-3 K=${ATREX_CHALLENGER_COUNT}"
echo "Per-Branch topology: Y=${ATREX_TRAJECTORIES_PER_BRANCH}, X=${ATREX_ATTEMPTS_PER_TRAJECTORY}"
echo "Local Runtime secrets: ${atrex_env_file}"
echo "Provider and Agate credentials remain only in the exported environment."
