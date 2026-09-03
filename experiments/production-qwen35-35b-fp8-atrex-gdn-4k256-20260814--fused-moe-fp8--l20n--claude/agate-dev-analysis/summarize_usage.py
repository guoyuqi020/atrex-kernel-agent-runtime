"""Describe the archived Dev requests; filename families are not cost buckets."""
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def summarize(folder):
    rows = json.loads(gzip.decompress((folder/'retained-dev.json.gz').read_bytes()))
    aka = json.loads(gzip.decompress((folder/'aka-dev-profile.json.gz').read_bytes()))
    counts, dsl, families = Counter(), defaultdict(Counter), Counter()
    known_families = (
        'bench_moe_latency.py','paired_ab_probe.py','ab_harness.py','harness_compare.py',
        'stage_ab_probe.py','fwd_latency_ab_probe.py','moe_bitwise_ab_probe.py',
        'moe_check_probe.py','probe_moe_correctness.py','moe_ood_probe.py','moe_stress_probe.py',
    )
    reuse = defaultdict(lambda: {'attempts':set(),'jobs':set(),'digests':set(),'epochs':set()})
    for row in rows:
        commands = {c['request'].get('command','') for c in row['calls']}
        if not commands:
            counts['unlinked'] += 1
            continue
        if len(commands) != 1:
            counts['ambiguous_command'] += 1
            continue
        counts['linked_unique_command'] += 1
        command = next(iter(commands))
        request = row['calls'][0]['request']
        files = request.get('file_paths') or []
        roots = {p.split('/',1)[0] for p in files}
        for key in ('scratch','tools'):
            if key in roots:
                counts['uploads_'+key] += 1
                dsl[row['dsl']]['uploads_'+key] += 1
        if not files:
            counts['no_file_paths'] += 1
        for name in known_families:
            if re.search(r'(?<![\w])'+re.escape(name)+r'(?![\w])',command):
                families[name] += 1
                dsl[row['dsl']][name] += 1
        if re.search(r'\bncu\b|profile_nvidia|collect_ncu|profile_driver',command):
            counts['explicit_ncu_command'] += 1
        for file in row['files']:
            if file['requested_path'].startswith('tools/'):
                item=reuse[row['dsl']+'/'+file['requested_path']]
                item['attempts'].add(row['attempt_id']);item['epochs'].add(row['epoch'])
                item['jobs'].add(row['job_id']);item['digests'].add(file['sha256'])
    return {
        'retained_linked_request_counts':dict(counts),
        'retained_script_families_calls_overlap':dict(families),
        'retained_by_dsl':dict(dsl),
        'retained_tools_reuse_final_archive_snapshots':{
            k:{'attempts':len(v['attempts']),'epochs':sorted(v['epochs']),
               'jobs':len(v['jobs']),'final_content_hashes':len(v['digests'])}
            for k,v in sorted(reuse.items(),key=lambda kv:-len(kv[1]['jobs'])) if len(v['attempts'])>1},
        'aka_explicit_ncu_or_driver_commands':dict(Counter(r['requested_kind'] for r in aka
            if re.search(r'\bncu\b|profile_nvidia|collect_ncu|profile_driver',' '.join(r['remote_argv_lexical'])))),
        'aka_by_dsl':{dsl:{'explicit_dev':sum(r['dsl']==dsl and r['requested_kind']=='dev' for r in aka),
            'profile':sum(r['dsl']==dsl and r['requested_kind']=='profile' for r in aka),
            'observed_fallback':sum(r['dsl']==dsl and r['fallback_to_dev'] for r in aka)} for dsl in ('cuda','triton','cutedsl')},
    }


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();value=summarize(args.data)
    (args.data/'usage-patterns.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False,indent=2))
