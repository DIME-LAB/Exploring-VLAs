"""Authoritative primitive cup geometry, in the rigid cup's BASE frame (metres)."""
import hashlib,json,math
from pathlib import Path
from pose_math import rotate,multiply

ROOT=Path(__file__).resolve().parents[3]
SOURCE='isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json'
RAW=(ROOT/SOURCE).read_bytes();SPEC=json.loads(RAW);CUP=SPEC['cup']
PRIMITIVES=CUP['primitives'];WALLS=[p for p in PRIMITIVES if 'interior_plane' in p]
# Live rest-contact penetration was 7–9 um. Allow 9 um observed + 10 um
# solver variability + 1 um float/transform error; only 20% of contact offset.
# Rest offsets are zero; this does NOT permit a 0.1 mm wall overlap.
WALL_REST_TOLERANCE_M=20e-6
FLOOR_REST_TOLERANCE_M=20e-6
RIM_CLEARANCE_M=CUP['contact_offset_m']
PROFILE=dict(version=2,kind='compound_box_halfplanes',source=SOURCE,source_sha256=hashlib.sha256(RAW).hexdigest(),
    frame='cup-base',drop_origin_offset_m=[0.,0.,CUP['rim_m']/2],primitives=PRIMITIVES,
    wall_rest_tolerance_m=WALL_REST_TOLERANCE_M,floor_rest_tolerance_m=FLOOR_REST_TOLERANCE_M,
    rim_clearance_m=RIM_CLEARANCE_M,expected_offsets_m=dict(cup_contact=CUP['contact_offset_m'],cup_rest=CUP['rest_offset_m'],block_contact=CUP['contact_offset_m'],block_rest=CUP['rest_offset_m']),
    offsets_basis='Authored primitive collision offsets; fresh tensor readback is required, not historical measurements',
    tolerance_basis='9 um measured rest penetration + 10 um solver variability + 1 um transform/float bound')
if (SPEC['version']!=1 or len(PRIMITIVES)!=13 or len(WALLS)!=12 or CUP['rest_offset_m']!=0 or
        len({p['name'] for p in PRIMITIVES})!=13):raise ValueError('Unexpected primitive cup schema')
for shape in PRIMITIVES:
    values=shape['center_m']+shape['size_m']+shape['quaternion_wxyz']
    if shape['type']!='box' or not all(math.isfinite(v) for v in values) or any(v<=0 for v in shape['size_m']):raise ValueError('Invalid primitive cup box')
    if abs(sum(v*v for v in shape['quaternion_wxyz'])-1)>1e-10:raise ValueError('Invalid cup box quaternion')
for wall in WALLS:
    plane=wall['interior_plane'];n=plane['normal']
    if len(n)!=3 or not all(math.isfinite(v) for v in n+[plane['offset_m']]) or sum(v*v for v in n)<=0:raise ValueError('Invalid cup wall plane')


def wall_checks(corners):
    checks=[]
    for index,point in enumerate(corners):
        for wall in WALLS:
            plane=wall['interior_plane'];n=plane['normal'];length=math.sqrt(sum(v*v for v in n))
            # n is intentionally NOT normalized in the asset. Divide the WHOLE
            # signed residual by ||n|| to express clearance in physical metres.
            distance=(plane['offset_m']-sum(a*b for a,b in zip(n,point)))/length
            checks.append(dict(corner=index,wall=wall['name'],height_m=point[2],clearance_m=distance,
                tolerance_m=WALL_REST_TOLERANCE_M,passed=distance>=-WALL_REST_TOLERANCE_M))
    return checks


def primitive_world_poses(drop):
    q=[drop.get(k,default) for k,default in zip(('qw','qx','qy','qz'),(1.,0.,0.,0.))]
    norm=math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or norm<1e-10:raise ValueError('Invalid cup pose quaternion')
    q=[v/norm for v in q];center=[drop[k] for k in ('x','y','z')]
    offset=rotate(q,[0,0,CUP['rim_m']/2]);base=[a-b for a,b in zip(center,offset)]
    return [dict(name=shape['name'],dimensions=list(shape['size_m']),
        position=[a+b for a,b in zip(base,rotate(q,shape['center_m']))],
        orientation=multiply(q,shape['quaternion_wxyz'])) for shape in PRIMITIVES]


def _pose(co,pose,snapshot):
    pos=[getattr(pose.position,k) for k in ('x','y','z')];q=[getattr(pose.orientation,k) for k in ('w','x','y','z')]
    parent=getattr(co,'pose',None)
    if parent is not None:
        pq=[getattr(parent.orientation,k) for k in ('w','x','y','z')]
        if sum(v*v for v in pq)<1e-15:pq=[1,0,0,0]
        pos=[a+getattr(parent.position,k) for a,k in zip(rotate(pq,pos),('x','y','z'))];q=multiply(pq,q)
    if co.header.frame_id=='world':
        base=snapshot['base_world_pose'];inv=[base['orientation'][0]]+[-v for v in base['orientation'][1:]]
        pos=rotate(inv,[a-b for a,b in zip(pos,base['position'])]);q=multiply(inv,q)
    elif co.header.frame_id not in ('base','/World/base'):raise ValueError('Unsupported cup planning frame')
    norm=math.sqrt(sum(v*v for v in q))
    if not math.isfinite(sum(pos)+norm) or norm<1e-10:raise ValueError('Invalid primitive pose')
    return pos,[v/norm for v in q]


def compound_pose_check(co,drop,snapshot):
    """Tolerance-aware bijection couples every box's shape to its physical pose."""
    if co is None or co.meshes or len(co.primitives)!=13 or len(co.primitive_poses)!=13:
        return dict(passed=False,error='Exactly thirteen primitive cup boxes required')
    expected=primitive_world_poses(drop);actual=[]
    try:
        for shape,pose in zip(co.primitives,co.primitive_poses):
            p,q=_pose(co,pose,snapshot);actual.append(dict(type=shape.type,dimensions=list(shape.dimensions),position=p,orientation=q))
    except ValueError as exc:return dict(passed=False,error=str(exc))
    edges=[]
    for wanted in expected:
        possible=[]
        for i,got in enumerate(actual):
            if got['type']!=1 or len(got['dimensions'])!=3 or not all(math.isfinite(v) for v in got['dimensions']):continue
            dim=max(abs(a-b) for a,b in zip(wanted['dimensions'],got['dimensions']))
            pos=max(abs(a-b) for a,b in zip(wanted['position'],got['position']))
            quat=min(max(abs(a-b) for a,b in zip(wanted['orientation'],got['orientation'])),max(abs(a+b) for a,b in zip(wanted['orientation'],got['orientation'])))
            if all(math.isfinite(v) for v in (dim,pos,quat)) and dim<=1e-7 and pos<=1e-5 and quat<=1e-5:possible.append(i)
        edges.append(possible)
    matched={}
    def assign(index,seen):
        for candidate in edges[index]:
            if candidate in seen:continue
            seen.add(candidate)
            if candidate not in matched or assign(matched[candidate],seen):matched[candidate]=index;return True
        return False
    passed=all(assign(i,set()) for i in range(len(expected)))
    return dict(passed=passed,shapes=len(actual),matching=matched,source_sha256=PROFILE['source_sha256'],
                expected_frame='cup-base',drop_center_offset_m=CUP['rim_m']/2,position_tolerance_m=1e-5,dimension_tolerance_m=1e-7)
