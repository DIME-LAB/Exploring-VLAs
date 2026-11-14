#!/usr/bin/env python3
"""Validate finalized copied collection batches without ROS, Isaac or CUDA."""
import argparse,hashlib,json,os,time
from pathlib import Path
from export_content import validate_metadata,ContentMismatch
from storage import atomic_json,confined


def input_signature(run):
    files=[run/'dataset/meta/collection_episodes.json',run/'summary.json',run/'collector-finalization.json',
        run/'dataset/meta/info.json',run/'dataset/meta/tasks.parquet',
        *sorted((run/'dataset/meta/episodes').rglob('*.parquet')),*sorted((run/'dataset/data').rglob('*.parquet')),
        *sorted(run.glob('attempt-*/frames.jsonl')),*sorted(run.glob('attempt-*/receipt.json')),
        *sorted(run.glob('terminal*')),*sorted(run.glob('attempt-*/terminal*'))]
    return [(str(f.relative_to(run)),f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns) for f in files]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pattern',default='rtx6000-normal-*')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=30.)
    args=p.parse_args();root=confined(args.root);output=confined(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    result=dict(version=1,mode='all_metadata',cpu_only=True,batches={},videos=dict(status='pending',policy='Two full episodes per batch; decoded-content pass follows separately'))
    result['validator_source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('validate_exports.py','export_content.py')}
    checked={}
    while True:
        for batch in sorted(root.glob(args.pattern)):
            run=batch/'normal';ready=run/'dataset/meta/collection_episodes.json'
            if not ready.exists():result['batches'][batch.name]=dict(status='PENDING',reason='No finalized dataset manifest');continue
            # Finished source batches are immutable; invalidate the cached pass
            # if their manifest, raw summary or table metadata changes.
            try:signature=input_signature(run)
            except FileNotFoundError:
                result['batches'][batch.name]=dict(status='PENDING',reason='Finalized batch transfer is missing an input');continue
            if checked.get(batch.name)==signature:continue
            print(json.dumps(dict(event='validation_start',batch=batch.name)),flush=True)
            try:
                value=validate_metadata(run);status='PASS'
                if signature!=input_signature(run):
                    raise RuntimeError('Source files changed while validating; retry after transfer settles')
                checked[batch.name]=signature
            except Exception as exc:
                status='FAIL';value=dict(passed=False,error=repr(exc),field=getattr(exc,'field',None),details=getattr(exc,'details',None))
            result['batches'][batch.name]=dict(status=status,run=str(run),metadata=value)
            result.update(updated_wall_ns=time.time_ns(),passed=all(x['status']=='PASS' for x in result['batches'].values()))
            atomic_json(output,result)
            print(json.dumps(dict(event='validation_done',batch=batch.name,status=status,episodes=value.get('episodes'),frames=value.get('frames'),wall_s=value.get('wall_s'),error=value.get('error'))),flush=True)
        result.update(updated_wall_ns=time.time_ns(),passed=bool(result['batches']) and all(x['status']=='PASS' for x in result['batches'].values()))
        result['production_episodes_checked']=sum(x.get('metadata',{}).get('episodes',0) for n,x in result['batches'].items() if 'smoke' not in n and x['status']=='PASS')
        result['training_launch_ready']=result['passed'] and result['production_episodes_checked']>=900
        atomic_json(output,result)
        if not args.watch:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
