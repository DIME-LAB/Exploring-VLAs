"""Seal clean labels and actual collector-applied commands separately."""
from pathlib import Path
import argparse,hashlib,json,math
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
rows=[json.loads(l) for l in (a.source/'episodes.jsonl').read_text().splitlines()];accepted={r['params']['seed']:r for r in rows if r['accepted']};episodes=[]
for seed in range(530000,530003):
 receipt=accepted[seed];file=a.source/f"attempt-{receipt['params']['trial']:04d}/frames.jsonl";raw=file.read_bytes();frames=[json.loads(l) for l in raw.splitlines()]
 initial=frames[0];episodes.append(dict(seed=seed,color=receipt['params']['color'],task=receipt['params']['task'],source_frames=str(file),source_sha256=hashlib.sha256(raw).hexdigest(),
  initial=dict(state=initial['physics_state'],drive_targets=initial['physics_drive_targets'],bodies=initial['physical_bodies'],robot_base_world_pose=initial['robot_base_world_pose']),
  clean_actions_rad=[f['action'] for f in frames],executed_actions_rad=[f['applied_command'] for f in frames],recorded_states_rad=[f['physics_state'] for f in frames],recorded_drive_targets_rad=[f['physics_drive_targets'] for f in frames],recorded_sim_ns=[f['sim_ns'] for f in frames],phases=[f['phase'] for f in frames],
  clean_executed_max_delta_deg=max(abs(x-y)*180/math.pi for f in frames for x,y in zip(f['action'],f['applied_command']))))
assert [r['color'] for r in episodes]==['red','green','blue'];a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(kind='native-held-replay-v1',source='rtx6000-normal-i3-b00',episodes=episodes),indent=2)+'\n');print(hashlib.sha256(a.out.read_bytes()).hexdigest())
