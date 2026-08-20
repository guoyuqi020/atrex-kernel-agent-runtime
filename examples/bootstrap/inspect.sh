#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
atrex_example_load_local_secrets

if [[ ! -f "${atrex_bootstrap_result_file}" ]]; then
  echo "Bootstrap result not found: ${atrex_bootstrap_result_file}" >&2
  echo "Run examples/bootstrap/bootstrap.sh first." >&2
  exit 66
fi
echo "Bootstrap result:"
"${atrex_python}" -m json.tool "${atrex_bootstrap_result_file}"
