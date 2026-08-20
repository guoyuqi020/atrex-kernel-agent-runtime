#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

if (( $# != 0 )); then
  echo "usage: $0" >&2
  exit 64
fi

atrex_example_prepare_gpu_wiki
atrex_example_require_remote_agate
atrex_example_require_agent_backend optimizer
atrex_example_require_agent_backend evolver
atrex_example_load_local_secrets
atrex_example_prepare_inputs
atrex_example_with_runtime bash "${script_dir}/run-campaign.sh" --inputs-already-prepared

echo
bash "${script_dir}/inspect.sh"
