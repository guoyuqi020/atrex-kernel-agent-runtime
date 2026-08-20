#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_gpu

arguments=(
  check
  --url "${AGATE_URL}"
  --gpu "${AGATE_GPU}"
  --deps-mode freeze_installed
  --wait
)
if [[ -n "${AGATE_ARCH:-}" ]]; then
  arguments+=(--arch "${AGATE_ARCH}")
fi
if [[ -n "${AGATE_SANITIZE:-}" ]]; then
  arguments+=(--sanitize "${AGATE_SANITIZE}")
fi
arguments+=("${agate_vecadd_candidate}")
exec "${agate_bin}" "${arguments[@]}"
