#!/usr/bin/env bash
# repro-loop.sh [N] [PARALLEL] — 用触发过 cuda.core 掉 shape 的原始 kernel 字节,
# 向 agate 反复提交单 shape(58) eval,每个 job 是一次独立 pod 抽签。
#
# 复现判据(与 2026-09-07 生产事故一致):
#   job.status == "succeeded"            <- 基础设施故障被误分类
#   result.passed.compile.58.status == "failed"
#   reason 含 ModuleNotFoundError: No module named 'cuda.core' (kernel.py:80)
#   correctness/performance == "skipped" ("Skipped because compile stage failed.")
#   result.environment.driver_version == "580.159.03"   <- 坏 pod 队列
#
# 用法:  ./repro-loop.sh 50 8      # 50 次抽签,8 并发
set -u
cd "$(dirname "$0")"
N=${1:-50}
PAR=${2:-8}
mkdir -p draws

draw() {
  i=$1
  key="repro-cudacore-$$-$i-$RANDOM"
  python3 make_payload.py --reference-dir reference-s58 \
      --idempotency-key "$key" --out "draws/draw-$i.json" >/dev/null || {
    echo "draw $i: PAYLOAD BUILD FAILED"; return; }
  agate submit "draws/draw-$i.json" --wait --wait-timeout 1800 \
      > "draws/result-$i.json" 2> "draws/stderr-$i.log"
  python3 - "$i" <<'PYEOF'
import json, sys
i = sys.argv[1]
try:
    d = json.load(open(f'draws/result-{i}.json'))
except Exception as e:
    print(f'draw {i}: UNPARSED RESULT ({e}) — see draws/stderr-{i}.log')
    sys.exit()
job = d.get('job') if isinstance(d.get('job'), dict) else d
res = job.get('result') or {}
env = res.get('environment') or {}
comp = (res.get('passed') or {}).get('compile') or {}
reason = ' '.join(str(v.get('reason') or '') for v in comp.values() if isinstance(v, dict))
st = [v.get('status') for v in comp.values() if isinstance(v, dict)]
hit = "No module named 'cuda.core'" in reason
tag = 'REPRO=YES' if hit else 'repro=no '
print(f"draw {i}: job={job.get('job_id')} status={job.get('status')} "
      f"driver={env.get('driver_version')} compile={st} {tag}")
PYEOF
}
export -f draw

seq 1 "$N" | xargs -P "$PAR" -I{} bash -c 'draw "$@"' _ {} | tee draws/summary.txt
echo "---"
echo "total draws : $(grep -c '^draw' draws/summary.txt)"
echo "reproduced  : $(grep -c 'REPRO=YES' draws/summary.txt)"
echo "driver mix  : $(grep -o 'driver=[^ ]*' draws/summary.txt | sort | uniq -c | tr '\n' ' ')"
