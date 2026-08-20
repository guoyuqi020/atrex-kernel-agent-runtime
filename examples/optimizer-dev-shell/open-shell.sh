#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

inputs_already_prepared=false
shell_name="zsh"
for argument in "$@"; do
  case "${argument}" in
    --inputs-already-prepared)
      inputs_already_prepared=true
      ;;
    zsh|bash)
      shell_name="${argument}"
      ;;
    *)
      echo "usage: $0 [--inputs-already-prepared] [zsh|bash]" >&2
      exit 64
      ;;
  esac
done

atrex_example_require_remote_agate
atrex_example_load_local_secrets
if [[ "${inputs_already_prepared}" == false ]]; then
  atrex_example_prepare_inputs
elif [[ ! -f "${atrex_config_file}" || ! -f "${atrex_campaign_file}" ]]; then
  echo "prepared Runtime config or Campaign definition is missing" >&2
  exit 66
fi
atrex_example_require_runtime_health

echo
echo "Opening a disposable Optimizer dev shell."
echo "No Campaign, Lineage, Bootstrap, or Agent backend will be started."
echo "Exit the shell to revoke its capability and destroy all temporary state."
exec "${atrex_runtime_cli}" temporary-dev-shell \
  --config "${atrex_config_file}" \
  --campaign "${atrex_campaign_file}" \
  --shell "${shell_name}"
