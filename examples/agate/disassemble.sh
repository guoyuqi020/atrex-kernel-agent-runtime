#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_gpu

exec "${agate_bin}" disassemble \
  --url "${AGATE_URL}" \
  --gpu "${AGATE_GPU}" \
  --candidate "${agate_vecadd_candidate}" \
  --fmt "${AGATE_DISASSEMBLY_FORMAT:-auto}" \
  --deps-mode freeze_installed \
  --wait
