#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

evolution_require_result
if ! command -v "${atrex_runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${atrex_runtime_cli}" >&2
  exit 69
fi
if [[ ! -f "${atrex_bootstrap_result_file}" || ! -f "${atrex_config_file}" ]]; then
  echo "Bootstrap result or Runtime config is missing from ${atrex_state_dir}." >&2
  exit 66
fi
selected_lineage_id="$(atrex_example_lineage_id)"

echo "Three-Epoch result:"
"${atrex_python}" -m json.tool "${evolution_result_file}"
echo
echo "Epoch winners:"
"${atrex_runtime_cli}" list-epochs \
  --config "${atrex_config_file}" \
  --lineage "${selected_lineage_id}" \
  --format table
echo
echo "Attempt history:"
"${atrex_runtime_cli}" list-attempts \
  --config "${atrex_config_file}" \
  --lineage "${selected_lineage_id}" \
  --format table
echo
echo "Kernel history:"
"${atrex_runtime_cli}" list-kernels \
  --config "${atrex_config_file}" \
  --lineage "${selected_lineage_id}" \
  --format table
echo
echo "Agent history:"
"${atrex_runtime_cli}" list-agent-revisions \
  --config "${atrex_config_file}" \
  --lineage "${selected_lineage_id}" \
  --format table
