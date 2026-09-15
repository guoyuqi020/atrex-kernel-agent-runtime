#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:-upstream-p128}"
case "$mode" in
  upstream-p128)
    remote_command="python3 smoke.py upstream-p128"
    note="验证 AKA FA4 R0 原始 HD256 2CTA/P128 内核可在 L20D 编译执行"
    ;;
  target)
    shape_id="${2:-0}"
    remote_command="python3 smoke.py target --shape-id ${shape_id}"
    note="验证 AKA FA4 R0 对当前 C05 Atrex-bench ABI 的 bring-up 进展"
    ;;
  *)
    echo "usage: $0 upstream-p128 | target [shape-id]" >&2
    exit 2
    ;;
esac

agate dev "$remote_command" \
  --gpu L20D \
  --working-dir . \
  --intent custom_harness \
  --note "$note" \
  --job-timeout 1800 \
  --wait-timeout 2100
