#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_gpu

exec "${agate_bin}" eval \
  --url "${AGATE_URL}" \
  --http-timeout "${AGATE_HTTP_TIMEOUT:-1800}" \
  --job-timeout "${AGATE_JOB_TIMEOUT:-3600}" \
  --wait-timeout "${AGATE_WAIT_TIMEOUT:-3900}" \
  --poll "${AGATE_POLL_SECONDS:-5}" \
  --gpu "${AGATE_GPU}" \
  --candidate "${agate_vecadd_candidate}" \
  --reference-dir "${agate_vecadd_reference}" \
  --operator vector_add \
  --num-correctness-cases "${AGATE_CORRECTNESS_CASES:-1}" \
  --bench-iters "${AGATE_BENCH_ITERS:-100}" \
  --mode "${AGATE_MODE:-full}" \
  --deps-mode freeze_installed \
  --lock-clocks \
  --wait
