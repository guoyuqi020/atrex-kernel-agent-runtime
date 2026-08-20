#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_load_env
atrex_demo_require_optimizer_backend

config_path="$(atrex_demo_config_path)"
campaign_file="${atrex_demo_example_dir}/campaign.json"
runtime_cli="${atrex_demo_runtime_cli}"
if [[ ! -f "${config_path}" ]]; then
  echo "Runtime config not found: ${config_path}" >&2
  exit 66
fi
if [[ ! -f "${campaign_file}" ]]; then
  echo "Campaign definition not found: ${campaign_file}" >&2
  exit 66
fi
if ! command -v "${runtime_cli}" >/dev/null 2>&1; then
  echo "Runtime CLI not found in PATH: ${runtime_cli}" >&2
  exit 69
fi

mkdir -p "$(dirname -- "${atrex_demo_bootstrap_result_file}")"
chmod 700 "$(dirname -- "${atrex_demo_bootstrap_result_file}")"
temporary="${atrex_demo_bootstrap_result_file}.tmp.$$"
trap 'rm -f "${temporary}"' EXIT
umask 077
"${runtime_cli}" bootstrap \
  --config "${config_path}" \
  --campaign "${campaign_file}" | tee "${temporary}"
mv "${temporary}" "${atrex_demo_bootstrap_result_file}"
trap - EXIT

lineage_id="$(atrex_demo_last_lineage_id)"
echo
echo "Bootstrap result saved to: ${atrex_demo_bootstrap_result_file}"
echo "Default lineage: ${lineage_id}"
echo "Open its Agent debug shell with:"
echo "  bash examples/local-wiki/start-agent-shell.sh"
