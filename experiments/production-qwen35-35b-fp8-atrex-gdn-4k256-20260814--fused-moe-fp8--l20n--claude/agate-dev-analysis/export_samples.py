"""Copy selected historical text as inert examples; never run the copied code."""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from audit_dev import extract_tools


RETAINED = {
    'retained-triton-benchmark':'dv_90d9b19d1939',
    'retained-cutedsl-paired-ab':'dv_4f1a85c8d14c',
    'retained-cuda-stage-timing':'dv_f0e52965ac28',
    'retained-cuda-stress':'dv_10d614158991',
    'retained-cuda-mma':'dv_c974fccb4944',
    'retained-cutedsl-inspection':'dv_01873c1f76b8',
}
AKA = {
    'aka-cutedsl-paired-ab':('AKA-1','cutedsl',13,1001,[
        'profiles/episode_13/harness/paired_time_driver.py',
        'profiles/episode_13/harness/incumbent_kernel.py']),
    'aka-triton-ncu':('AKA-1','triton',15,976,[
        'profiles/episode_15/harness/collect_ncu.sh']),
    'aka-triton-upload-failure':('AKA-2','triton',14,354,[
        'profiles/episode_14/harness/mma_roofline_probe.py']),
}


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def copy_file(archive, original, destination, manifest, kind):
    original=original.resolve()
    if not original.is_relative_to(archive):
        raise ValueError('Source path escaped archive')
    if not original.is_file():
        manifest.append({'source':str(original.relative_to(archive)), 'missing':True,'kind':kind})
        return
    payload=original.read_bytes()
    payload.decode('utf-8')
    if len(payload)>512*1024:
        raise ValueError('Unexpected large sample file')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(payload)
    manifest.append({'source':str(original.relative_to(archive)),
        'copy':destination.name,'kind':kind,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})


def replay_helper(archive, trace, before_line, relative, runtime):
    """Literal Write/Edit replay only, with conservative invalidation for Shell edits.

    Never eval Python/Shell. Unknown mutation means no reconstructed snapshot.
    """
    content=None;lines=[]
    for tool in extract_tools((archive/trace).read_bytes(),runtime):
        if tool['line']>=before_line:
            break
        if any(r['is_error'] for r in tool['results']):
            continue
        inp=tool['input'];path=inp.get('file_path','')
        matches=path==relative or path.endswith('/'+relative)
        if matches and tool['name']=='Write':
            content=inp.get('content');lines=[tool['line']]
        elif matches and tool['name']=='Edit':
            old=inp.get('old_string','');new=inp.get('new_string','')
            if content is not None and old and old in content:
                content=content.replace(old,new) if inp.get('replace_all') else content.replace(old,new,1)
                lines.append(tool['line'])
            else:
                content=None;lines=[]
        elif tool['name']=='Bash':
            command=inp.get('command','')
            if Path(relative).name in command and re.search(r'\bsed\b|\bperl\b|\bcp\b|\bmv\b|write_text|write_bytes|\.write\(|\bpatch\b|cat\s*>',command):
                content=None;lines=[]
    return (content,lines) if content is not None else (None,[])


def export_replayed(archive, folder, trace, before_line, relative, runtime, manifest):
    content,lines=replay_helper(archive,trace,before_line,relative,runtime)
    if content is None:
        return
    destination=folder/'at-call'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(content)
    manifest.append({'copy':str(destination.relative_to(folder)),'kind':'literal_trace_reconstruction',
        'trace':trace,'write_edit_lines':lines,'before_call_line':before_line,
        'sha256':hashlib.sha256(content.encode()).hexdigest(),'bytes':len(content.encode())})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();archive=args.archive_root.resolve();root=args.output.resolve()
    if root.is_relative_to(archive):
        raise ValueError('Never write examples into the original archive')
    retained=json.loads(gzip.decompress((args.data/'retained-dev.json.gz').read_bytes()))
    aka=json.loads(gzip.decompress((args.data/'aka-dev-profile.json.gz').read_bytes()))
    for name,job in RETAINED.items():
        row=next(r for r in retained if r['job_id']==job)
        assert len(row['calls'])==1, (name,row['calls'])
        call=row['calls'][0];folder=root/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'command.txt').write_text(call['command']+'\n')
        manifest=[]
        copy_file(archive,archive/call['request_path'],folder/'request.json',manifest,'final_request_file')
        for entry in row['files']:
            relative=Path(entry['requested_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid sample file path')
            copy_file(archive,archive/entry['archive_path'],folder/'files'/relative,manifest,'final_workspace_snapshot')
            export_replayed(archive,folder,row['trace'],call['line'],str(relative),True,manifest)
        # The Kernel and Gateway Result are immutable per-operation artifacts.
        state=archive/'runtime/workspace-full-20260902.unpacked/production/control-l20n/state/artifacts/sha256'
        kernel=state/row['kernel_artifact_digest'].split(':')[1]/'payload/kernel.py'
        copy_file(archive,kernel,folder/'files/work/kernel/kernel.py',manifest,'immutable_kernel_artifact')
        result=state/row['gateway_result_digest'].split(':')[1]/'payload/value.json'
        copy_file(archive,result,folder/'result.json',manifest,'immutable_gateway_result')
        meta={k:row[k] for k in ('dsl','epoch','trajectory','iteration','attempt_id','job_id','trace','workspace','kernel_artifact_digest','gateway_result_digest')}
        meta.update(trace_line=call['line'],result_lines=call['direct_result_lines'],files=manifest,
            snapshot_note='Request and helper files are end-of-Attempt snapshots, not guaranteed per-call upload bytes. Kernel/result are frozen artifacts. command.txt is original Trace text. Examples are evidence, not runnable production recipes.')
        write_json(folder/'provenance.json',meta)
        (folder/'stdout.txt').write_text(row['stdout']+'\n')
    for name,(instance,dsl,episode,line,files) in AKA.items():
        row=next(r for r in aka if (r['instance'],r['dsl'],r['episode'],r['line'])==(instance,dsl,episode,line))
        folder=root/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'command.txt').write_text(row['command']+'\n')
        (folder/'output.txt').write_text(row['output']+'\n')
        stem='atrex-runs' if instance=='AKA-1' else 'atrex-runs2'
        episode_root=archive/'AKA'/(stem+'.with-traces')/stem/('kernel_opt_fused_moe_fp8_'+dsl+'_l20n_production')/'.atrex_long_horizon/episodes'/f'e{episode:04d}'
        manifest=[]
        for relative in files:
            copy_file(archive,episode_root/'archive/worktree_files'/relative,folder/'files'/relative,manifest,'final_episode_snapshot')
            export_replayed(archive,folder,row['trace'],line,relative,False,manifest)
        meta={k:row[k] for k in ('instance','dsl','episode','trace','line','output_lines','requested_kind','observed_routes','dev_job_ids')}
        meta.update(files=manifest,snapshot_note='Commands/results are exact visible Trace text. Helper files are end-of-Episode snapshots; not a claim of identical content at every invocation. AKA framework files and remote injected inputs are not reconstructed.')
        write_json(folder/'provenance.json',meta)
    print(f'Exported {len(RETAINED)+len(AKA)} offline examples to {root}')


if __name__=='__main__':
    main()
