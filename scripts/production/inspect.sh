#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

workspace=""
selected_dsl=""
while (( $# > 0 )); do
  case "$1" in
    --workspace) workspace="${2:-}"; shift 2 ;;
    --dsl) selected_dsl="${2:-}"; shift 2 ;;
    *) echo "usage: $0 --workspace DIR [--dsl cuda|triton|cutedsl]" >&2; exit 64 ;;
  esac
done
if [[ -z "${workspace}" ]]; then
  echo "usage: $0 --workspace DIR [--dsl cuda|triton|cutedsl]" >&2
  exit 64
fi
if [[ -n "${selected_dsl}" ]]; then
  case "${selected_dsl}" in cuda|triton|cutedsl) ;; *) echo "Invalid DSL: ${selected_dsl}" >&2; exit 64 ;; esac
fi
atrex_prod_workspace_paths "${workspace}"
atrex_prod_require_workspace
atrex_prod_require_policy_gate
atrex_prod_load_environment ""
atrex_prod_require_commands
echo "Production content policy gate: enabled"

dsls=(cuda triton cutedsl)
[[ -n "${selected_dsl}" ]] && dsls=("${selected_dsl}")
for dsl in "${dsls[@]}"; do
  atrex_prod_dsl_paths "${dsl}"
  echo
  echo "================ ${dsl} ================"
  echo "Workspace: ${atrex_prod_dsl_workspace}"
  if [[ ! -f "${atrex_prod_bootstrap_result}" ]]; then
    echo "Bootstrap result not found: ${atrex_prod_bootstrap_result}"
    continue
  fi
  campaign_id="$(atrex_prod_json_value "${atrex_prod_bootstrap_result}" campaign_id)"
  echo "Campaign: ${campaign_id}"
  echo
  "${atrex_prod_cli}" list-epochs --config "${atrex_prod_config}" \
    --campaign "${campaign_id}" --format table
  echo
  "${atrex_prod_cli}" list-kernels --config "${atrex_prod_config}" \
    --campaign "${campaign_id}" --format table
  echo
  "${atrex_prod_cli}" list-agent-revisions --config "${atrex_prod_config}" \
    --campaign "${campaign_id}" --format table
done
