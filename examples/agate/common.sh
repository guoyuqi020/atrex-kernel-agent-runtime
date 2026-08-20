#!/usr/bin/env bash

agate_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agate_runtime_root="$(cd -- "${agate_example_dir}/../.." && pwd)"
agate_bin="${AGATE_BIN:-agate}"
agate_vecadd_root="${agate_runtime_root}/examples/shared/vecadd"
agate_vecadd_candidate="${agate_vecadd_root}/triton/agate-candidate/kernel.py"
agate_vecadd_reference="${agate_vecadd_root}/reference"

agate_require_service() {
  if ! command -v "${agate_bin}" >/dev/null 2>&1; then
    echo "Agate CLI not found in PATH: ${agate_bin}" >&2
    echo "Activate a platform-local environment containing agate or set AGATE_BIN." >&2
    return 69
  fi
  if [[ -z "${AGATE_URL:-}" ]]; then
    echo "AGATE_URL must name the real Agate service for this example." >&2
    return 64
  fi
}

agate_require_gpu() {
  agate_require_service
  if [[ -z "${AGATE_GPU:-}" ]]; then
    echo "AGATE_GPU must be an exact environment returned by 'agate env'." >&2
    return 64
  fi
}
