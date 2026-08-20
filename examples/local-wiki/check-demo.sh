#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo-common.sh
source "${script_dir}/demo-common.sh"
atrex_demo_require_optimizer_backend

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the demo readiness checks" >&2
  exit 69
fi
check_endpoint() {
  local label="$1"
  local url="$2"
  curl --fail --silent --show-error --max-time 10 "${url}" >/dev/null
  echo "ready: ${label} (${url})"
}

check_endpoint "Local Wiki health" "http://127.0.0.1:8091/healthz"
check_endpoint "Local Wiki corpus" "http://127.0.0.1:8091/readyz"
check_endpoint "Runtime health" "http://127.0.0.1:8765/healthz"
check_endpoint "Runtime dependencies" "http://127.0.0.1:8765/readyz"

echo "Local Wiki example prerequisites are ready."
