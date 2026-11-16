"""Seal the first fifteen recorded training homes; never runs a simulator."""
from pathlib import Path
import argparse,collections,hashlib,json,math,os,tempfile
CUT='bf7336d93d5ba054c5edc66cea685725eb30eb40'
COLLECTOR_SHA='b74cced3bca9425b55bc033d6803489685b71832d22cec5ab2828e6dda307e31'
SELECTION=tuple((batch,index,seed) for batch,count,start in (('b00',6,530000),('b01',6,530500),('b02',3,531000)) for index,seed in enumerate(range(start,start+count)))
ROOT=Path('/mnt/a4500-backup/exploring-vlas')

def digest(path):
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed '+str(path))
    return h.hexdigest()

def first_frame(path):
    before=path.stat();h=hashlib.sha256();first=None;count=0
    with path.open('rb') as stream:
        for line in stream:
            h.update(line);count+=1
            if first is None:first=json.loads(line)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Frames changed')
    return first,count,h.hexdigest()

def check_frame(frame,source):
    stamp=frame['sim_ns']
    if not all(frame[k]==stamp for k in ('observation_stamp_ns','top_stamp_ns','wrist_stamp_ns')):raise ValueError('First frame camera/state join differs')
    if abs(stamp-source['sim_ns']-1e9/30)>1000:raise ValueError('Training first frame is not one native tick after preparation')
    if len(frame['state'])!=6 or not all(math.isfinite(x) for x in frame['state']):raise ValueError('Bad first state')
    delta=max(abs(math.degrees(a-b)) for a,b in zip(frame['state'],source['state']))
    if delta>.01:raise ValueError('First-frame arm moved beyond reset tolerance')
    return delta

def make_bank(audit_path,dataset_root,out):
    assert dataset_root.resolve().is_relative_to(ROOT/'datasets')
    assert out.resolve().is_relative_to(ROOT/'eval') and not out.exists()
    audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['all_backup_shas_match'] and audit['episodes']==900
    entries=audit['entries'][:15];assert [e['episode_index'] for e in entries]==list(range(15))
    pending=Path(tempfile.mkdtemp(prefix='.'+out.name+'-incoming-',dir=out.parent))
    files={};sources={};references={};colors=[]
    def write(name,value):
        p=pending/name;p.write_text(json.dumps(value,indent=2)+'\n');files[name]=digest(p)
    for global_index,((batch,index,seed),entry) in enumerate(zip(SELECTION,entries)):
        batch='rtx6000-normal-i3-'+batch;attempt=dataset_root/batch/'normal'/f'attempt-{index:04d}'
        assert entry['seed']==seed and entry['source_batch']==batch
        assert Path(entry['videos']['observation.images.top']['source']).parent==attempt
        receipt_path=attempt/'receipt.json';frames_path=attempt/'frames.jsonl';run_path=attempt.parent/'run.json'
        receipt_sha=digest(receipt_path);receipt=json.loads(receipt_path.read_text());assert digest(receipt_path)==receipt_sha
        run_sha=digest(run_path);run=json.loads(run_path.read_text());assert digest(run_path)==run_sha
        assert run['git_commit']==CUT and run['fps']==30 and run['source_sha256']['linux-env/scripts/collect/collect.py']==COLLECTOR_SHA
        params=receipt['params'];color=params['color'];colors.append(color)
        assert receipt['accepted'] and params['recovery']=='none' and params['seed']==seed and receipt['episode_id']==entry['episode_id'] and color==entry['color']
        assert params['task']==f'Pick a {color} lego and place it in {color} cup'
        start=receipt['episode_start_physical'];frame,count,frames_sha=first_frame(frames_path);assert count==entry['frames']
        delta=check_frame(frame,start);assert len(start['state'])==len(start['drive_targets'])==6
        body={name:{key:value[key] for key in ('position','orientation')} for name,value in start['bodies'].items()}
        assert set(body)=={f'{c}_2x3_{i}' for c in ('red','green','blue') for i in range(2)}|{'cup_red','cup_green','cup_blue'}
        state=dict(seed=seed,color=color,task=params['task'],mode='home',frame_index=0,global_episode_index=global_index,source_batch=batch,state=start['state'],recorded_drive_targets=start['drive_targets'],state_source='Recorded native episode_start_physical',robot_base_world_pose=start['robot_base_world_pose'],frame_sim_ns=frame['sim_ns'],source_frames=str(frames_path),source_frames_sha256=frames_sha,source_receipt=str(receipt_path),source_receipt_sha256=receipt_sha,reference_run=str(run_path),reference_run_sha256=run_sha,training_first_frame_offset_ns=frame['sim_ns']-start['sim_ns'],first_frame_joint_delta_deg=delta)
        write(f'state-{seed}.json',state);write(f'seed-{seed}.json',body)
        sources[str(seed)]=dict(global_episode_index=global_index,source_batch=batch,receipt_sha256=receipt_sha,frames_sha256=frames_sha,reference_run_sha256=run_sha,first_frame_joint_delta_deg=delta)
        references[batch]=dict(path=str(run_path),sha256=run_sha)
    assert collections.Counter(colors)==dict(red=5,green=5,blue=5)
    manifest=dict(kind='next120-seen-training-home',format_version=1,complete=True,mode='home',seeds=[s for _,_,s in SELECTION],global_episode_indices=list(range(15)),colour_counts=dict(collections.Counter(colors)),source_audit_path=str(audit_path),source_audit_sha256=digest(audit_path),reference_runs=references,sources=sources,files_sha256=files,producer_sha256=digest(Path(__file__)))
    p=pending/'bank-manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n')
    for name,sha in files.items():assert digest(pending/name)==sha
    assert not out.exists();os.rename(pending,out)
    return dict(status='CPU-SEALED-NOT-LIVE-QUALIFIED',path=str(out),sha256=digest(out/'bank-manifest.json'),seeds=manifest['seeds'],colour_counts=manifest['colour_counts'],reference_runs=references)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--datasets',type=Path,default=ROOT/'datasets');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(make_bank(a.audit,a.datasets,a.out)),flush=True)
