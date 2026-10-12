"""Launch two independent upstream AKA CUDA campaigns without changing AKA code."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/workspace')
BASE = ROOT / 'runs'
BASE.mkdir(exist_ok=True)
lock = (BASE / '.launch.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
record = BASE / 'launched.json'
if record.exists():
    raise SystemExit('Launch record exists; inspect processes before explicitly recovering')
env = os.environ.copy()
env.update(json.loads((ROOT / '.auth/agate.json').read_text()))
env.update(json.loads((Path.home() / '.claude/settings.json').read_text()).get('env', {}))
env.update(IS_SANDBOX='1', AGATE_URL='https://atrex-gateway.alibaba-inc.com', AGATE_GPU='ZW-M890P')
records = []
for op in json.loads((ROOT / 'experiment.json').read_text())['operators']:
    name = op['operator']
    target = BASE / name
    target.mkdir(exist_ok=True)
    command = ['python3', str(ROOT / 'aka/orchestrator/optimize.py'), '--op-dir', str(ROOT / 'atrex-bench/operators' / name), '--platform', 'ZW-M890P', '--sandbox-hardware', 'ZW-M890P', '--sandbox-profile', 'prod', '--framework', 'Cuda', '--max-iters', '15', '--workspace', str(target)]
    with (target / 'orchestrator.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT/'aka', env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    records.append({'operator':name,'pid':proc.pid,'command':command,'started_at':time.time()})
record.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
