#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
atrex_example_require_remote_agate
atrex_example_require_agent_backend optimizer
atrex_example_load_local_secrets
atrex_example_prepare_inputs

echo "Local Runtime secrets: ${atrex_env_file}"
echo "Agate credentials remain only in the exported environment."
