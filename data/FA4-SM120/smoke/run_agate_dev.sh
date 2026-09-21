#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:-sm120-bf16}"
case "$mode" in
  sm120-bf16)
    remote_command="python3 smoke.py sm120-bf16"
    note="验证 FA4 R0 的 SM120 BF16 Dense Kernel 可在 L20N 编译执行"
    ;;
  target)
    shape_id="${2:-0}"
    remote_command="python3 smoke.py target --shape-id ${shape_id}"
    note="验证 FA4 SM120 R0 对 FP8 paged-KV 生产 ABI 的正确性"
    ;;
  *)
    echo "usage: $0 sm120-bf16 | target [shape-id]" >&2
    exit 2
    ;;
esac

agate dev "$remote_command" \
  --gpu L20N \
  --working-dir . \
  --intent custom_harness \
  --note "$note" \
  --job-timeout 1800 \
  --wait-timeout 2100
