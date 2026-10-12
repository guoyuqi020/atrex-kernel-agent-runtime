import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/workspace');env=os.environ.copy()
env.update(json.loads((Path.home()/'.claude/settings.json').read_text()).get('env',{}))
env.update(IS_SANDBOX='1')
checks={}
p=subprocess.run(['claude','--print','Reply with exactly AKA_READY','--output-format','json','--max-turns','1','--tools',''],capture_output=True,text=True,env=env,timeout=120)
try:
 response=json.loads(p.stdout)
 checks['claude']={'returncode':p.returncode,'is_error':response.get('is_error'),'result':response.get('result'),'model_usage':list(response.get('modelUsage',{}))}
except ValueError:
 checks['claude']={'returncode':p.returncode,'error':'Non-JSON response; inspect smoke-private.log privately'}
request={'request':'ZW-M890P PPU hardware bandwidth and CUDA memory coalescing techniques','max_records':3,'max_bytes':12000}
p=subprocess.run(['python3',str(root/'aka/plugins/gpu-wiki/internal_wiki_adapter.py')],input=json.dumps(request),capture_output=True,text=True,env=env,timeout=360,cwd=root/'aka')
if p.returncode:
 checks['wiki']={'returncode':p.returncode,'error':p.stderr[-2000:]}
else:
 d=json.loads(p.stdout);checks['wiki']={'returncode':0,'query_id':d.get('query_id'),'records':list(d.get('records',{})),'notes':d.get('notes')}
 (root/'wiki-smoke-result.json').write_text(json.dumps(d,indent=2)+'\n')
checks['checked_at']=time.time()
(root/'smoke-result.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
