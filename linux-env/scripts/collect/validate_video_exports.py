#!/usr/bin/env python3
"""Sample two complete raw/export video episodes per already-checked batch."""
import argparse,hashlib,json,time
from pathlib import Path
from export_video import validate_videos,selected_episodes,POLICY
from storage import atomic_json,confined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=30.)
    args=p.parse_args();metadata=confined(args.metadata);output=confined(args.output)
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    result=dict(version=1,cpu_only=True,policy=POLICY,batches={});checked={}
    result['validator_source_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('validate_video_exports.py','export_video.py','export_content.py')}
    while True:
        source=json.loads(metadata.read_text())
        for name,batch in sorted(source['batches'].items()):
            if batch['status']!='PASS':continue
            run=Path(batch['run']);meta=batch['metadata']
            meta_sha=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            files=[*sorted((run/'dataset/videos').rglob('*.mp4')),*sorted((run/'dataset/meta/episodes').rglob('*.parquet'))]
            for i in selected_episodes(run,meta['episode_details'],2):
                files.extend(run/meta['episode_details'][i]['raw_attempt']/(c+'.mp4') for c in ('top','wrist'))
            def signature():return [(str(f),f.stat().st_ino,f.stat().st_size,f.stat().st_mtime_ns) for f in files]
            try:before=signature()
            except FileNotFoundError:
                result['batches'][name]=dict(status='PENDING',reason='Video transfer missing a file');continue
            key=(meta_sha,before)
            if checked.get(name)==key:continue
            print(json.dumps(dict(event='video_start',batch=name)),flush=True)
            try:
                value=validate_videos(run,meta,sample_count=2)
                if before!=signature():raise RuntimeError('Video files changed during validation')
                checked[name]=key;status='PASS'
            except Exception as exc:
                status='FAIL';value=dict(passed=False,error=str(exc),field=getattr(exc,'field',None),details=getattr(exc,'details',None))
            result['batches'][name]=dict(status=status,metadata_sha256=meta_sha,validation=value)
            result.update(updated_wall_ns=time.time_ns(),passed=all(v['status']=='PASS' for v in result['batches'].values()))
            atomic_json(output,result)
            print(json.dumps(dict(event='video_done',batch=name,status=status,wall_s=value.get('wall_s'),error=value.get('error'))),flush=True)
        atomic_json(output,result)
        if not args.watch:break
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
