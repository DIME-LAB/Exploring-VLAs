"""Describe native base_link movement without changing the articulation or scene."""
import math


def angle(a,b):
    if len(a)!=4 or len(b)!=4 or not all(math.isfinite(x) for x in a+b):raise ValueError('Invalid native base quaternion')
    na=math.sqrt(sum(x*x for x in a));nb=math.sqrt(sum(x*x for x in b))
    if min(na,nb)<1e-12:raise ValueError('Zero native base quaternion')
    return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b))/(na*nb))))


def summary(initial,rows):
    present=[r for r in rows if 'robot_base_world_pose' in r]
    if not present:return dict(available=False,reason='No recorded native base poses')
    if len(present)!=len(rows) or not initial or 'robot_base_world_pose' not in initial:raise ValueError('Incomplete native base drift sequence')
    start=initial['robot_base_world_pose'];translations=[];angles=[]
    if len(start['position'])!=3 or not all(math.isfinite(v) for v in start['position']):raise ValueError('Invalid initial base position')
    for r in rows:
        if r.get('robot_base_pose_source')!='physx_rigid_body':raise ValueError('Base pose is not measured from PhysX')
        p=r['robot_base_world_pose']
        if len(p['position'])!=3 or not all(math.isfinite(v) for v in p['position']):raise ValueError('Invalid base position')
        translations.append(math.dist(start['position'],p['position']));angles.append(angle(start['orientation'],p['orientation']))
    return dict(available=True,source='PhysX RigidPrim base_link at each held tick',frames=len(rows),
        initial=start,max_translation_m=max(translations),max_rotation_rad=max(angles),
        final_translation_m=translations[-1],final_rotation_rad=angles[-1])
