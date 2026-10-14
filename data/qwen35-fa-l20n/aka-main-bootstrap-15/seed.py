"""Import the sealed Bootstrap into an upstream AKA workspace without editing AKA."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/workspace')
sys.path.insert(0, str(ROOT / 'aka'))
from orchestrator.campaign import Campaign
from orchestrator.constants import WORKSPACE_INIT
from orchestrator.hardware import framework_workspace_suffix
from orchestrator.operator_layout import agent_visible_operator_files
from orchestrator.session_io import _record_local_test_result
from orchestrator.workspace_state import git_head

definition = json.loads((ROOT / 'experiment/experiment.json').read_text())
operator = ROOT / 'atrex-bench/operators' / definition['operator']
baseline = ROOT / 'experiment/bootstrap'
runroot = ROOT / 'runs'
runroot.mkdir(exist_ok=True)
campaign = Campaign(
    name=definition['operator'], kernel_demo=str(operator/'reference.py'),
    platform='L20N', framework='Cuda', arch='sm_120', work_dir=str(runroot),
    workspace_suffix=framework_workspace_suffix('Cuda', 'L20N', 'leaderboard'),
    sandbox_hardware='L20N', sandbox_profile='prod',
    atrex_bench_root=str(ROOT/'atrex-bench'), max_iters=15,
)
w = campaign.workspace
assert not w.exists(), 'A workspace already exists; refuse to replace experiment progress'
subprocess.run(['bash', str(WORKSPACE_INIT), campaign.campaign_name, str(baseline/'kernel.py')], cwd=runroot, check=True)
for name in agent_visible_operator_files(operator, generalized=False):
    source=operator/name
    if source.is_file():shutil.copy2(source,w/name)
campaign._link_runtime()
campaign._install_native_evaluator()
campaign._install_profile_driver()
readme=campaign._native_v0_readme(generalized=False).replace(
    'V0 is the verbatim PyTorch reference wrapper; optimized versions must',
    'V0 is the imported, previously evaluated CUDA Bootstrap kernel; optimized versions must')
(w/'README.md').write_text(readme)
provenance=json.loads((baseline/'provenance.json').read_text())
assert hashlib.sha256((w/'kernel.py').read_bytes()).hexdigest()==provenance['kernel_sha256']
(w/'bootstrap-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
sources=['.gitignore','CLAUDE.md','README.md','kernel.py','test_kernel.py','profile_driver.py','bootstrap-provenance.json',*agent_visible_operator_files(operator,generalized=False)]
sources=list(dict.fromkeys(p for p in sources if (w/p).is_file()))
subprocess.run(['git','add',*sources],cwd=w,check=True)
subprocess.run(['git','commit','-m','V0: import original evaluated GDN Bootstrap source'],cwd=w,check=True)
source_commit=git_head(w)
result=json.loads((baseline/'result.json').read_text())
memory_path=_record_local_test_result(w,'v0',result)
memory=json.loads(memory_path.read_text())
memory['git_commit_hash']=source_commit
memory['optimization']={'action_category':'baseline','action_description':'Imported original Runtime CUDA Bootstrap and its sealed 90-shape Agate measurements; no new measurement performed during import.','expected_impact':'common initial kernel for independent repeats','risks_and_rollback':'historical measurement; evaluate new candidates with upstream AKA gates'}
memory['source_bootstrap']=provenance
memory_path.write_text(json.dumps(memory,indent=2)+'\n')
(w/'baseline_report.md').write_text('# Imported Bootstrap baseline\n\nThe original CUDA source and successful official Agate measurements were imported unchanged from the sealed Runtime Bootstrap. This import does not rerun measurement.\n\nSource hash: '+provenance['kernel_sha256']+'\n\n90 shapes; geometric mean '+str(result['latency_us_geomean'])+' us. Source jobs and contract hashes are recorded in bootstrap-provenance.json.\n')
subprocess.run(['git','add','memory/v0.json','baseline_report.md'],cwd=w,check=True)
subprocess.run(['git','commit','-m','V0: record imported Bootstrap evidence'],cwd=w,check=True)
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=w,text=True).strip(), 'Seed workspace has uncommitted state'
assert len(memory['performance']['latency_us_by_shape'])==90
assert memory['correctness']['status']=='PASS' and memory['quality_gate']['result']=='PASS'
record={'workspace':str(w),'kernel_sha256':provenance['kernel_sha256'],'source_commit':source_commit,'bootstrap_latency_us':result['latency_us_geomean'],'shape_count':90,'aka_commit':definition['aka_commit']}
(runroot/'seeded.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
