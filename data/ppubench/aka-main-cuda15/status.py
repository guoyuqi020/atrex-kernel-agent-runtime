import json
from pathlib import Path
root=Path('/workspace')
out=[]
p=root/'runs/launched.json'
for item in json.loads(p.read_text()) if p.exists() else []:
    proc=Path('/proc')/str(item['pid'])
    try:
        status=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
    except FileNotFoundError:
        status='exited'
    run=Path(item['command'][-1])
    memories=[]
    for path in run.glob('kernel_opt_*/memory/v*.json'):
        d=json.loads(path.read_text())
        memories.append({'file':str(path.relative_to(run)),'version':d.get('version'),'keys':list(d)})
    out.append({'operator':item['operator'],'pid':item['pid'],'process_state':status,'log_tail':(run/'orchestrator.log').read_text()[-1800:],'memory':memories})
print(json.dumps(out,indent=2))
