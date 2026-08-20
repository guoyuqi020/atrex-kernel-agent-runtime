#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_service

echo "Resolved Agate connection:"
"${agate_bin}" config --url "${AGATE_URL}"
echo
echo "Agate health:"
"${agate_bin}" health --url "${AGATE_URL}"
echo
echo "Selectable GPU environments:"
"${agate_bin}" env --url "${AGATE_URL}" --json
