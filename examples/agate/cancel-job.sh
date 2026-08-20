#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"
agate_require_service
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <job-id>" >&2
  exit 64
fi

exec "${agate_bin}" cancel --url "${AGATE_URL}" "$1"
