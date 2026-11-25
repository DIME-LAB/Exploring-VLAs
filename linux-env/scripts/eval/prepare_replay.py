"""Extract exact recorded label actions and states for one episode per colour."""
from pathlib import Path
import hashlib,json,math,argparse
BASE=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval')
p=argparse.ArgumentParser();p.add_argument('--starts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve().is_relative_to(BASE) and not a.out.exists()
plan=json.loads((a.starts/'home-plan.json').read_text());episodes={}
for index,row in enumerate(plan['rows'][:3]):
 state=json.loads((Path(plan['bank'])/f"state-{row['seed']}.json").read_text());file=Path(state['source_frames']);raw=file.read_bytes();assert hashlib.sha256(raw).hexdigest()==state['source_frames_sha256'];frames=[json.loads(l) for l in raw.splitlines()]
 episodes[f'ep{index:02d}']=dict(row,actions_degrees=[[math.degrees(q) for q in f['action']] for f in frames],recorded_state_rad=[f['physics_state'] for f in frames],recorded_drive_targets_rad=[f['physics_drive_targets'] for f in frames],recorded_sim_ns=[f['sim_ns'] for f in frames],recorded_base=frames[0]['robot_base_world_pose'],source_frames_sha256=state['source_frames_sha256'],source_frames=str(file),label='exact dataset action, including recorded DART offsets; no policy/model inference')
assert [r['color'] for r in episodes.values()]==['red','green','blue'];a.out.write_text(json.dumps(episodes,indent=2)+'\n');print(json.dumps(dict(episodes=3,path=str(a.out),sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))
