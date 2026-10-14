import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/workspace');sys.path.insert(0,str(ROOT/'aka'))
from orchestrator.campaign import Campaign
from orchestrator.constants import WORKSPACE_INIT
from orchestrator.hardware import framework_workspace_suffix
from orchestrator.operator_layout import agent_visible_operator_files
from orchestrator.session_io import _record_local_test_result
from orchestrator.workspace_state import git_head
d=json.loads((ROOT/'experiment/experiment.json').read_text());operator=ROOT/'atrex-bench/operators'/d['operator'];baseline=ROOT/'experiment/bootstrap';runroot=ROOT/'runs';runroot.mkdir(exist_ok=True)
c=Campaign(name=d['operator'],kernel_demo=str(operator/'reference.py'),platform='L20D',framework='Cuda',arch='sm_103',work_dir=str(runroot),workspace_suffix=framework_workspace_suffix('Cuda','L20D','leaderboard'),sandbox_hardware='L20D',sandbox_profile='prod',atrex_bench_root=str(ROOT/'atrex-bench'),max_iters=15)
w=c.workspace;assert not w.exists(),'workspace already exists'
subprocess.run(['bash',str(WORKSPACE_INIT),c.campaign_name,str(baseline/'source/kernel.py')],cwd=runroot,check=True)
for name in agent_visible_operator_files(operator,generalized=False):
 s=operator/name
 if s.is_file():shutil.copy2(s,w/name)
for p in (baseline/'source').rglob('*'):
 if p.is_file():
  q=w/p.relative_to(baseline/'source');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
c._link_runtime();c._install_native_evaluator();c._install_profile_driver()
readme=c._native_v0_readme(generalized=False).replace('V0 is the verbatim PyTorch reference wrapper; optimized versions must','V0 is the imported, previously evaluated common GDN Bootstrap source tree; optimized versions must')
(w/'README.md').write_text(readme)
prov=json.loads((baseline/'provenance.json').read_text());assert hashlib.sha256((w/'kernel.py').read_bytes()).hexdigest()==prov['kernel_sha256']
(w/'bootstrap-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
subprocess.run(['git','add','.'],cwd=w,check=True);subprocess.run(['git','commit','-m','V0: import common evaluated GDN Bootstrap source'],cwd=w,check=True)
source_commit=git_head(w);result=json.loads((baseline/'result.json').read_text());assert result.get('correct') is True and result.get('correctness',{}).get('status')=='PASS';result['all_pass']=True;mp=_record_local_test_result(w,'v0',result);m=json.loads(mp.read_text());m['git_commit_hash']=source_commit;m['optimization']={'action_category':'baseline','action_description':'Imported the common Runtime GDN Bootstrap source tree and sealed 10-shape Agate measurements; no new measurement performed during import.','expected_impact':'identical initial kernel for three independent upstream AKA repeats','risks_and_rollback':'historical measurement; all new candidates use upstream AKA evaluation gates'};m['source_bootstrap']=prov;mp.write_text(json.dumps(m,indent=2)+'\n')
(w/'baseline_report.md').write_text('# Imported common Bootstrap baseline\n\nThe exact Runtime Bootstrap source tree and successful official 10-shape Agate measurement were imported unchanged for all three repeats.\n\nKernel SHA256: '+prov['kernel_sha256']+'\n\nGeometric mean: '+str(result['latency_us_geomean'])+' us.\n')
subprocess.run(['git','add','memory/v0.json','baseline_report.md'],cwd=w,check=True);subprocess.run(['git','commit','-m','V0: record imported common Bootstrap evidence'],cwd=w,check=True)
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=w,text=True).strip();assert len(m['performance']['latency_us_by_shape'])==10;assert m['correctness']['status']=='PASS' and m['quality_gate']['result']=='PASS'
record={'workspace':str(w),'kernel_sha256':prov['kernel_sha256'],'kernel_artifact_digest':prov['kernel_artifact_digest'],'source_commit':source_commit,'bootstrap_latency_us':result['latency_us_geomean'],'shape_count':10,'aka_commit':d['aka_commit']};(runroot/'seeded.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
