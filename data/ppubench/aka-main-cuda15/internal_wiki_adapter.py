"""Connect AKA's existing GPU Wiki tool contract to the pinned internal indexed store."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

root = Path(__file__).resolve().parents[2] / 'internal_gpu_wiki'
request = json.load(sys.stdin)
with tempfile.TemporaryDirectory(prefix='aka-internal-wiki-') as directory:
    path = Path(directory) / 'request.txt'
    path.write_text(request['request'], encoding='utf-8')
    sys.argv = [str(root/'tools/query_nl.py'), '--store-root', str(root), '--file', str(path)]
    for field in ('max_records', 'max_bytes', 'exclude'):
        if field in request:
            sys.argv.extend(['--'+field.replace('_','-'), str(request[field])])
    os.environ.pop('ATREX_WIKI_STORE_ROOT', None)
    sys.path.insert(0, str(root/'tools'))
    runpy.run_path(sys.argv[0], run_name='__main__')
