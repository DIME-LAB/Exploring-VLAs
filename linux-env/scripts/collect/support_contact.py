"""Micrometre support contact is allowed only at the measured segment start."""
import math
from cup_geometry import CUP,WALLS,FLOOR_REST_TOLERANCE_M
from pose_math import rotate
from scene_sync import physical_pose_caches

POLICY=dict(version=1,depth_max_m=FLOOR_REST_TOLERANCE_M,position_error_m=1e-6,
    normal_min_cos=.999,source='measured segment start only; targets and swept interior remain strict',
    basis='9 um observed rest residual + 10 um solver variability + 1 um transform/float bound; contact=0.1 mm, rest=0')
LINKS={'base','shoulder','upper_arm','lower_arm','wrist','gripper','jaw','camera_mount','usb_camera','camera_link','tcp_link'}


def contact_rows(contacts):
    rows=[]
    for c in contacts:
        row=dict(body_1=c.contact_body_1,body_2=c.contact_body_2,depth_m=float(c.depth))
        if hasattr(c,'position'):row['position']=[float(getattr(c.position,k)) for k in ('x','y','z')]
        if hasattr(c,'normal'):row['normal']=[float(getattr(c.normal,k)) for k in ('x','y','z')]
        if hasattr(c,'header'):row['frame_id']=c.header.frame_id
        rows.append(row)
    return rows


def context(snapshot,ground_z,attached=None):
    _,cups=physical_pose_caches(snapshot,primitive_cups=True)
    return dict(frame_id='base',ground_z_m=ground_z,cups={'cup_'+k:v for k,v in cups.items()},
        attached_object='lego_'+attached if attached else None)


def classify(rows,ctx):
    proof=dict(passed=False,policy=POLICY,contacts=rows,context=ctx,allowed=[])
    if not rows:return proof
    for c in rows:
        depth=c['depth_m'];a,b=c['body_1'],c['body_2']
        if not math.isfinite(depth) or depth<0 or depth>POLICY['depth_max_m']:return proof
        if c.get('frame_id')!=ctx['frame_id'] or len(c.get('position',[]))!=3 or len(c.get('normal',[]))!=3:return proof
        p,n=c['position'],c['normal'];length=math.sqrt(sum(v*v for v in n))
        if not all(math.isfinite(v) for v in p+n) or length<.5:return proof
        robot=a if a in LINKS or a==ctx.get('attached_object') else b if b in LINKS or b==ctx.get('attached_object') else None
        if robot is None:return proof
        other=b if robot==a else a
        tolerance=POLICY['depth_max_m']+POLICY['position_error_m']
        if other=='ground_plane':
            if abs(p[2]-ctx['ground_z_m'])>tolerance or abs(n[2])/length<POLICY['normal_min_cos']:return proof
            # The authored ground is a 2m box; side/underside contacts are not support.
            if max(abs(p[0]),abs(p[1]))>=.99:return proof
            proof['allowed'].append(dict(kind='ground',**c));continue
        cup=ctx['cups'].get(other)
        if cup is None:return proof
        q=[cup[k] for k in ('qw','qx','qy','qz')];inv=[q[0],-q[1],-q[2],-q[3]]
        offset=rotate(q,[0,0,CUP['rim_m']/2]);base=[cup[k]-v for k,v in zip(('x','y','z'),offset)]
        local=rotate(inv,[a-b for a,b in zip(p,base)]);normal=rotate(inv,n)
        if abs(local[2]-CUP['floor_top_m'])>tolerance or abs(normal[2])/length<POLICY['normal_min_cos']:return proof
        for wall in WALLS:
            plane=wall['interior_plane'];v=plane['normal'];clearance=(plane['offset_m']-sum(a*b for a,b in zip(v,local)))/math.sqrt(sum(a*a for a in v))
            if clearance<0:return proof
        proof['allowed'].append(dict(kind='cup_floor',local_position=local,**c))
    proof['passed']=True
    return proof


def row_errors(row):
    errors=[]
    for index,path in enumerate(row.get('execution_geometry',{}).get('paths',[])):
        proof=path.get('measured_start_contact')
        if proof is None:continue
        try:
            if index!=0 or proof!=classify(proof['contacts'],proof['context']) or not proof['passed']:
                errors.append('Unverified or misplaced measured support contact')
        except (KeyError,ValueError,TypeError):errors.append('Incomplete measured support contact proof')
    return errors
