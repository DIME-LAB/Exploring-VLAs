"""Read-only, all-row raw-to-LeRobot table equivalence checks (CPU only)."""
import hashlib,json,time
from pathlib import Path
import numpy as np

VERSION=1
COLUMNS=('timestamp','frame_index','episode_index','index','task_index','observation.state','action')


class ContentMismatch(ValueError):
    def __init__(self,field,details):
        self.field=field;self.details=details
        super().__init__(field+': '+json.dumps(details,sort_keys=True,default=str))


def equal_column(name,actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=expected.shape:raise ContentMismatch(name,dict(actual_shape=actual.shape,expected_shape=expected.shape))
    same=actual==expected
    if not np.all(same):
        where=np.argwhere(~same)[0];index=tuple(where)
        raise ContentMismatch(name,dict(index=list(map(int,where)),actual=actual[index].item(),expected=expected[index].item()))


def read_parquet_rows(paths,columns=None):
    import pyarrow.parquet as pq
    return [row for path in paths for row in pq.read_table(path,columns=columns).to_pylist()]


def load_source(run):
    """Numeric order comes from attempt IDs; rejected attempts never enter export."""
    run=Path(run);summary=json.loads((run/'summary.json').read_text());final=json.loads((run/'collector-finalization.json').read_text())
    if (run/'superseded.json').exists() or not summary.get('complete') or summary.get('end_marker')!='COLLECTION_FINISHED':
        raise ContentMismatch('finalization',dict(reason='Raw run is incomplete or superseded'))
    if not final.get('completed') or not final.get('passed') or final.get('cleanup_errors'):
        raise ContentMismatch('finalization',dict(reason='Raw producer has no clean affirmative finalization'))
    if list(run.glob('terminal*')) or list(run.glob('attempt-*/terminal*')):
        raise ContentMismatch('finalization',dict(reason='Raw run has a terminal marker'))
    accepted=[]
    for path in sorted(run.glob('attempt-*/receipt.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
        receipt=json.loads(path.read_text())
        if receipt.get('accepted'):accepted.append((path.parent,receipt))
    if len(accepted)!=summary['episodes']:raise ContentMismatch('accepted count',dict(receipts=len(accepted),summary=summary['episodes']))
    return accepted


def validate_metadata(run):
    """Compare every row, measured timestamp, task and index with immutable raw."""
    started=time.monotonic();run=Path(run);root=run/'dataset'
    info=json.loads((root/'meta/info.json').read_text())
    if info['fps']!=30:raise ContentMismatch('fps',dict(actual=info['fps'],expected=30))
    accepted=load_source(run)
    manifest=json.loads((root/'meta/collection_episodes.json').read_text())
    expected_manifest=[dict(episode_index=i,**receipt) for i,(_,receipt) in enumerate(accepted)]
    if manifest!=expected_manifest:raise ContentMismatch('episode manifest',dict(reason='Stored receipts/order differ from accepted raw'))
    tasks=read_parquet_rows([root/'meta/tasks.parquet'])
    task_map={};expected_tasks=[]
    for _,receipt in accepted:
        task=receipt['params']['task']
        if task not in task_map:
            task_map[task]=len(task_map);expected_tasks.append(dict(task_index=task_map[task],task=task))
    if tasks!=expected_tasks:raise ContentMismatch('tasks table',dict(actual=tasks,expected=expected_tasks))
    paths=sorted((root/'data').rglob('*.parquet'))
    if not paths:raise ContentMismatch('data',dict(reason='No Parquet data files'))
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    data={k:[] for k in COLUMNS}
    for path in paths:
        table=pq.read_table(path,columns=list(COLUMNS),use_threads=False)
        for k in COLUMNS:data[k].extend(table[k].to_pylist())
    meta=read_parquet_rows(sorted((root/'meta/episodes').rglob('*.parquet')),
        ['episode_index','tasks','length','dataset_from_index','dataset_to_index'])
    if len(meta)!=len(accepted) or info['total_episodes']!=len(accepted):
        raise ContentMismatch('episode count',dict(meta=len(meta),info=info['total_episodes'],raw=len(accepted)))
    offset=0;episodes=[];raw_hashes={};max_timestamp_error_s=0.
    for episode,(path,receipt) in enumerate(accepted):
        raw=path/'frames.jsonl';h=hashlib.sha256();rows=[]
        with raw.open('rb') as stream:
            for line in stream:h.update(line);rows.append(json.loads(line))
        raw_hashes[str(raw.relative_to(run))]=h.hexdigest()
        if not rows:raise ContentMismatch('raw episode',dict(episode=episode,reason='No frames'))
        n=len(rows);end=offset+n;task=receipt['params']['task']
        if any(row.get('episode_id')!=receipt['episode_id'] for row in rows):raise ContentMismatch('raw episode identity',dict(episode=episode))
        stamps=np.asarray([r['sim_ns'] for r in rows],dtype=np.int64)
        if np.any(np.diff(stamps)<=0):raise ContentMismatch('raw clock',dict(episode=episode,reason='Non-monotonic'))
        times=(stamps-stamps[0])/1e9
        # The export column is float32. Its correctly rounded source value is
        # exact; no extra timing tolerance can conceal a shifted frame.
        equal_column('timestamp',data['timestamp'][offset:end],times.astype(np.float32))
        max_timestamp_error_s=max(max_timestamp_error_s,float(np.max(np.abs(np.asarray(data['timestamp'][offset:end])-times))))
        for key,expected in [('frame_index',np.arange(n)),('episode_index',np.full(n,episode)),
                ('index',np.arange(offset,end)),('task_index',np.full(n,task_map[task]))]:
            equal_column(key,data[key][offset:end],expected)
        for key in ('observation.state','action'):
            raw_key='state' if key=='observation.state' else 'action'
            expected=np.rad2deg(np.asarray([r[raw_key] for r in rows],dtype=np.float32))
            equal_column(key,data[key][offset:end],expected)
        expected_meta=dict(episode_index=episode,tasks=[task],length=n,dataset_from_index=offset,dataset_to_index=end)
        if meta[episode]!=expected_meta:raise ContentMismatch('episode index metadata',dict(actual=meta[episode],expected=expected_meta))
        episodes.append(dict(episode_index=episode,raw_attempt=path.name,seed=receipt['params']['seed'],task=task,frames=n,from_index=offset,to_index=end))
        offset=end
    if any(len(data[k])!=offset for k in COLUMNS) or info['total_frames']!=offset:
        raise ContentMismatch('frame count',dict(info=info['total_frames'],raw=offset,columns={k:len(v) for k,v in data.items()}))
    return dict(passed=True,version=VERSION,scope='ALL episodes: timestamps, frame/episode/global/task indices, task table, episode metadata, state/action and raw manifest',
        episodes=len(episodes),frames=offset,fps=info['fps'],episode_details=episodes,raw_frame_sha256=raw_hashes,
        timestamp_rule='exact correctly rounded float32 of measured raw relative sim time',max_float32_timestamp_rounding_s=max_timestamp_error_s,
        wall_s=time.monotonic()-started)
