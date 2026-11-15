"""Sealed, explicitly diagnostic starts from five accepted training episodes."""
from pathlib import Path
import argparse,hashlib,json,math,os
COLORS=('red','green','blue');SEEDS=tuple(range(530000,530005))
NAMES={f'{c}_2x3_{i}' for c in COLORS for i in range(2)}|{f'cup_{c}' for c in COLORS}
BASE=Path('/mnt/a4500-backup/exploring-vlas')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def validate_bank(bank,expected_sha):
 bank=Path(bank);manifest=bank/'bank-manifest.json'
 if digest(manifest)!=expected_sha:raise ValueError('Training-start bank SHA differs')
 m=json.loads(manifest.read_text())
 if m.get('kind')!='seen-training-start-diagnostic' or not m.get('complete') or m.get('seeds')!=list(SEEDS) or m.get('mode') not in ('home','mid'):raise ValueError('Invalid training-start diagnostic contract')
 expected={'baseline.json'}|{f'seed-{s}.json' for s in SEEDS}|{f'state-{s}.json' for s in SEEDS}
 if set(m['files_sha256'])!=expected:raise ValueError('Incomplete training-start files')
 for name,sha in m['files_sha256'].items():
  if digest(bank/name)!=sha:raise ValueError('Training-start file mutated: '+name)
 return m

def entry(bank,seed,expected_sha):
 m=validate_bank(bank,expected_sha)
 if seed not in SEEDS:raise ValueError('Unplanned seen-training seed')
 value=json.loads((Path(bank)/f'state-{seed}.json').read_text());bodies=json.loads((Path(bank)/f'seed-{seed}.json').read_text())
 q=value['state'];assert len(q)==6 and all(math.isfinite(v) for v in q)
 base=value.get('robot_base_world_pose',{})
 if len(base.get('position',[]))!=3 or len(base.get('orientation',[]))!=4 or not all(math.isfinite(v) for v in base['position']+base['orientation']):raise ValueError('Missing or malformed recorded physical base pose')
 assert set(bodies)==NAMES and value['seed']==seed and value['mode']==m['mode']
 assert value['color'] in COLORS and value['task']==f"Pick a {value['color']} lego and place it in {value['color']} cup"
 for body in bodies.values():
  assert len(body['position'])==3 and len(body['orientation'])==4 and all(math.isfinite(v) for v in body['position']+body['orientation'])
  assert abs(sum(v*v for v in body['orientation'])-1)<1e-4
 return dict(value,bodies=bodies)

def select_mid(frames):
 initial=frames[0]['physics_state'];depart=next((i for i,f in enumerate(frames) if max(abs(x-y) for x,y in zip(f['physics_state'][:5],initial[:5]))>math.radians(2)),None)
 if depart is None:raise ValueError('No arm departure in source episode')
 index=depart+30
 if index>=len(frames):raise ValueError('Source lacks one second after departure')
 if abs(frames[index]['sim_ns']-frames[depart]['sim_ns']-1_000_000_000)>2:raise ValueError('Source cadence differs from true30Hz')
 maximum=-math.inf
 for f in frames[:index+1]:
  maximum=max(maximum,f['physics_state'][5])
  if maximum-f['physics_state'][5]>=math.radians(2):raise ValueError('Mid start is after closure')
 return depart,index

def prepare(out):
 out=Path(out).resolve();assert out.is_relative_to(BASE/'eval') and not out.exists();out.mkdir()
 normal=BASE/'datasets/rtx6000-normal-i3-b00/normal';job=BASE/'train-data/phase3.2/jobs/smolvla-phase3-normal-60k-s42-20261114'
 training=json.loads((job/'batch-manifest.json').read_text());assert str(normal/'dataset') in {r['root'] for r in training['batches']}
 accepted={r['params']['seed']:r for r in map(json.loads,(normal/'episodes.jsonl').read_text().splitlines()) if r['accepted']};assert set(SEEDS)<=set(accepted)
 sources={};plans={mode:[] for mode in ('home','mid')}
 for mode in plans:(out/mode).mkdir()
 for seed in SEEDS:
  receipt=accepted[seed];assert receipt['params']['recovery']=='none';trial=receipt['params']['trial'];file=normal/f'attempt-{trial:04d}/frames.jsonl';frames=[json.loads(line) for line in file.open()]
  depart,index=select_mid(frames)
  for mode,i in (('home',0),('mid',index)):
   frame=frames[i];bodies={n:{k:frame['physical_bodies'][n][k] for k in ('position','orientation')} for n in sorted(NAMES)};folder=out/mode
   value=dict(seed=seed,color=receipt['params']['color'],task=receipt['params']['task'],mode=mode,frame_index=i,departure_frame=depart,sim_s_after_departure=(frame['sim_ns']-frames[depart]['sim_ns'])/1e9,
              state=frame['physics_state'],recorded_drive_targets=frame['physics_drive_targets'],state_source='recorded measured PhysX articulation',
              robot_base_world_pose=frame['robot_base_world_pose'],frame_sim_ns=frame['sim_ns'],source_frames=str(file),source_frames_sha256=digest(file),
              source_receipt=str(normal/f'attempt-{trial:04d}/receipt.json'),source_receipt_sha256=digest(normal/f'attempt-{trial:04d}/receipt.json'),
              reset_semantics='Zero velocities; controller and articulation drive targets held at measured state. Original momentum/drive lead is not resumed.')
   write(folder/f'seed-{seed}.json',bodies);write(folder/f'state-{seed}.json',value)
   if seed==SEEDS[0]:write(folder/'baseline.json',bodies)
   plans[mode].append(dict(seed=seed,color=value['color'],task=value['task'],rtc=True,suite='seen-training-'+mode,phrase='exact'))
  sources[str(seed)]=dict(frames_sha256=digest(file),receipt_sha256=digest(normal/f'attempt-{trial:04d}/receipt.json'))
 for mode,plan in plans.items():
  folder=out/mode;files=sorted(folder.glob('*.json'))
  manifest=dict(kind='seen-training-start-diagnostic',complete=True,mode=mode,seeds=list(SEEDS),colors=list(COLORS),files_sha256={p.name:digest(p) for p in files},source_batch='rtx6000-normal-i3-b00',sources=sources,producer_sha256=digest(__file__))
  write(folder/'bank-manifest.json',manifest);sha=digest(folder/'bank-manifest.json');write(out/(mode+'-plan.json'),dict(bank=str(folder),bank_sha256=sha,rows=plan));validate_bank(folder,sha)
 print(json.dumps(dict(prepared=True,out=str(out),seeds=SEEDS,modes=list(plans))),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare',));p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.out)
