import fcntl,json,os,subprocess,time,hashlib
from pathlib import Path
root=Path('/workspace');runs=root/'runs';lock=(runs/'.launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (runs/'launched.json').exists(),'already launched'
d=json.loads((root/'experiment/experiment.json').read_text());seed=json.loads((runs/'seeded.json').read_text());w=Path(seed['workspace'])
assert hashlib.sha256((w/'kernel.py').read_bytes()).hexdigest()==seed['kernel_sha256'];assert subprocess.check_output(['git','-C',str(root/'aka'),'rev-parse','HEAD'],text=True).strip()==d['aka_commit'];assert not subprocess.check_output(['git','-C',str(root/'aka'),'diff','--name-only','--ignore-submodules=all'],text=True).strip()
env=os.environ.copy();env.update(json.loads((root/'.auth/agate.json').read_text()));settings=json.loads(Path('/root/.claude/settings.json').read_text())['env'];assert settings.get('ANTHROPIC_MODEL')=='qwen3.8-max';env.update(settings);env.update(IS_SANDBOX='1',AGATE_URL='https://atrex-gateway.alibaba-inc.com',AGATE_GPU='L20D')
cmd=['python3',str(root/'aka/orchestrator/optimize.py'),'--op-dir',str(root/'atrex-bench/operators'/d['operator']),'--platform','L20D','--sandbox-hardware','L20D','--sandbox-profile','prod','--framework','Cuda','--max-iters','15','--workspace',str(runs),'--arch','sm_103']
with (runs/'orchestrator.log').open('ab',buffering=0) as log:p=subprocess.Popen(cmd,cwd=root/'aka',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':p.pid,'command':cmd,'started_at':time.time(),**seed};(runs/'launched.json').write_text(json.dumps(rec,indent=2)+'\n');time.sleep(5);assert p.poll() is None,'orchestrator exited';print(json.dumps(rec))
