#!/usr/bin/env python3
"""Post-hoc command-FK characterization using the production robot FK module."""
import argparse,csv,hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path);p.add_argument('--fk-module',type=Path,required=True)
a=p.parse_args();root=a.root.resolve()
spec=importlib.util.spec_from_file_location('production_fk',a.fk_module);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
rows=[]
inputs=sorted((root/'raw').glob('*/ep*/result.json'))
if not inputs:raise FileNotFoundError(f'No episode results under {root / "raw"}')
for resultfile in inputs:
 r=json.loads(resultfile.read_text());ep=resultfile.parent
 if r['outcome'] not in {'success','grasped-not-placed','no-grasp'}:raise ValueError(f'Invalid scored episode: {resultfile}')
 samples=[json.loads(line) for line in (ep/'poses.jsonl').read_text().splitlines()]
 samples=[s for s in samples if s['t']<=r['time_s']]
 if not samples:raise ValueError(f'No policy-window pose samples: {ep}')
 joints=np.asarray([[s['joints'][j] for j in fk.ARM_JOINT_NAMES] for s in samples]);tips=fk.forward_kinematics_batch(joints)
 if not np.isfinite(tips).all():raise ValueError(f'Non-finite FK: {ep}')
 grip=np.asarray([s['joints']['gripper_joint'] for s in samples]);gaps=.019+.0749*grip
 closest=(float('inf'),None,None)
 trace=[];maxlifts={};lift_since={};sustained=set()
 starts={k:v['position'][2] for k,v in r['randomize']['bodies'].items() if not k.startswith('cup_')}
 for i,(s,tip) in enumerate(zip(samples,tips)):
  blue=[(float(np.linalg.norm(tip-np.asarray(pos[:3]))),name) for name,pos in s['objects'].items() if name.startswith('blue')]
  d,name=min(blue)
  if d<closest[0]:closest=(d,i,name)
  for obj,z0 in starts.items():
   if obj not in s['objects']:continue
   lift=s['objects'][obj][2]-z0;maxlifts[obj]=max(maxlifts.get(obj,0),lift)
   lift_since[obj]=(lift_since.get(obj) if lift_since.get(obj) is not None else s['t']) if lift>=.024 else None
   if lift_since[obj] is not None and s['t']-lift_since[obj]>=.5:sustained.add(obj)
  trace.append(dict(time_s=s['t'],tcp_x_m=tip[0],tcp_y_m=tip[1],tcp_z_m=tip[2],nearest_blue=name,blue_center_distance_m=d,
                    gripper_deg=math.degrees(grip[i]),estimated_jaw_gap_m=gaps[i]))
 min_i=int(np.argmin(grip));running_max=np.maximum.accumulate(grip);closing=running_max-grip
 closure_indices=np.flatnonzero(closing>=math.radians(2));close_i=int(closure_indices[0]) if len(closure_indices) else min_i
 nearest_obj=min((float(np.linalg.norm(tips[close_i]-np.asarray(pos[:3]))),name) for name,pos in samples[close_i]['objects'].items())
 summary=dict(condition=r['condition'],episode=r['episode'],seed=r['seed'],outcome=r['outcome'],
  distance_basis='URDF tcp_link FK from recorded mock-controller reference; not measured physical TCP',
  closest_blue_center_distance_m=closest[0],closest_blue=closest[2],closest_time_s=samples[closest[1]]['t'],
  closest_tcp_xyz_m=tips[closest[1]].tolist(),
  closest_gripper_deg=math.degrees(float(grip[closest[1]])),closest_estimated_jaw_gap_m=float(gaps[closest[1]]),
  closing_motion_ge2deg=bool(len(closure_indices)),closing_motion_max_deg=math.degrees(float(closing.max())),
  gripper_min_deg=math.degrees(float(grip.min())),gripper_max_deg=math.degrees(float(grip.max())),
  estimated_min_jaw_gap_m=float(gaps.min()),
  closure_or_min_opening_time_s=samples[close_i]['t'],closure_tcp_xyz_m=tips[close_i].tolist(),
  closure_nearest_object=nearest_obj[1],closure_nearest_object_distance_m=nearest_obj[0],
  min_opening_time_s=samples[min_i]['t'],min_opening_tcp_xyz_m=tips[min_i].tolist(),
  min_opening_nearest_blue=trace[min_i]['nearest_blue'],min_opening_blue_distance_m=trace[min_i]['blue_center_distance_m'],
  any_sustained_lift_24mm=bool(sustained),sustained_lift_objects=sorted(sustained),
  max_any_object_lift_m=max(maxlifts.values()),max_lift_by_object_m=maxlifts,
  sampled_pose_count=len(samples),fk_source_sha256=hashlib.sha256(a.fk_module.read_bytes()).hexdigest())
 if sustained:mode='sustained lift: '+','.join(sorted(sustained))
 elif closest[0]>.05:mode='commanded TCP stayed >5cm from blue centres'
 elif not len(closure_indices):mode='nearer blue approach; <2deg closing motion'
 else:mode='nearer blue approach and closure; no sustained object lift'
 summary['failure_characterization']=mode
 (ep/'characterization.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (ep/'tcp_analysis.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
 rows.append(summary)
with (root/'failure-characterization.csv').open('w') as f:
 fields=[k for k in rows[0] if k not in {'max_lift_by_object_m'}]
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
(root/'failure-characterization.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(episodes=len(rows),closest_blue_range_m=[min(r['closest_blue_center_distance_m'] for r in rows),max(r['closest_blue_center_distance_m'] for r in rows)],closing_motion_ge2deg=sum(r['closing_motion_ge2deg'] for r in rows),any_lift=sum(r['any_sustained_lift_24mm'] for r in rows))))
