"""Install the knowledge-only adapter; leave AKA orchestration source unchanged."""
import json
from pathlib import Path
import shutil

root=Path('/workspace/aka')
p=root/'plugins/gpu-wiki/plugin.json'
d=json.loads(p.read_text())
d['tools']['query']['command']=['{python}','{plugin_root}/internal_wiki_adapter.py']
d['resources']={'gpu-wiki':'../../internal_gpu_wiki'}
shutil.copyfile('/workspace/internal_wiki_adapter.py',p.parent/'internal_wiki_adapter.py')
p.write_text(json.dumps(d,indent=2)+'\n')
print('gpu-wiki.query connected to pinned internal_gpu_wiki indexed native query')
