#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_gpu

command="${1:-nvidia-smi}"
exec "${agate_bin}" dev \
  --url "${AGATE_URL}" \
  --gpu "${AGATE_GPU}" \
  --intent inspect \
  --note "atrex-runtime Agate example" \
  --wait \
  "${command}"
