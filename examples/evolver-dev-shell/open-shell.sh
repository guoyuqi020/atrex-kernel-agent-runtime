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

if [[ "${inputs_already_prepared}" == false ]]; then
  export AGATE_URL="${AGATE_URL:-http://127.0.0.1:9}"
  export AGATE_GPU="${AGATE_GPU:-nvidia-h100}"
  atrex_example_prepare_inputs
elif [[ ! -f "${atrex_config_file}" || ! -f "${atrex_campaign_file}" ]]; then
  echo "prepared Runtime config or Campaign definition is missing" >&2
  exit 66
fi

echo "Opening a disposable Evolver dev shell."
echo "No Campaign, Lineage, Epoch, Bootstrap, Runtime service, or Agent backend will be started."
echo "Exit the shell to destroy all temporary state."
exec "${atrex_runtime_cli}" temporary-evolver-dev-shell \
  --config "${atrex_config_file}" \
  --campaign "${atrex_campaign_file}" \
  --shell "${shell_name}"
