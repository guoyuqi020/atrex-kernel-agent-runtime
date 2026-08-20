#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_ensure_env

echo "Local demo secrets are ready: ${atrex_demo_env_file}"
echo "Runtime, Bootstrap, and Agent Shell wrappers load this file automatically."
