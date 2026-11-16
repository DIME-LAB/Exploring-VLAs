"""Native bit identity and bounded triplet pose agreement, independent of clocks/labels."""
import copy,hashlib,json,math,numbers,struct

BODY_NAMES=tuple(sorted([f'{c}_2x3_{i}' for c in ('red','green','blue') for i in range(2)]+
                        ['cup_red','cup_green','cup_blue']))
SCHEMA='soarm101-native-physical-state-v1'
TRIPLET_POSE_POLICY=dict(version=1,position_m=1e-6,orientation_geodesic_deg=1e-3,
    arm_joint_rad=0.,drive_target_rad=0.,joint_velocity_rad_s=0.,body_velocity=0.,base_pose_bits_exact=True,
    strict='arm positions/velocities/drives, body velocities and both base poses retain exact native bits',
    scope='only body position/orientation comparisons across independent restores or member observations',
    quaternion='normalized shortest geodesic via atan2; exact bits retained diagnostically')


def vector(value,length,precision='f'):
    if len(value)!=length or any(not isinstance(x,numbers.Real) or isinstance(x,bool) or not math.isfinite(x) for x in value):
        raise ValueError('Incomplete or non-finite physical vector')
    return [struct.unpack('>'+precision,struct.pack('>'+precision,float(x)))[0] for x in value]


def physical_state(snapshot):
    """PhysX feedback is float32; the USD World/base transform is float64."""
    if set(snapshot['bodies'])!=set(BODY_NAMES):raise ValueError('Physical identity requires all nine bodies')
    out={k:vector(snapshot[k],6) for k in ('state','velocity','drive_targets')}
    out['bodies']={name:{key:vector(snapshot['bodies'][name][key],size)
        for key,size in (('position',3),('orientation',4),('velocity',6))} for name in BODY_NAMES}
    for key,precision in (('base_world_pose','d'),('robot_base_world_pose','f')):
        out[key]={name:vector(snapshot[key][name],size,precision) for name,size in (('position',3),('orientation',4))}
    return out


def identity(snapshot):
    data=physical_state(snapshot);bits=bytearray(SCHEMA.encode())
    def add(value,precision='f'):
        for x in value:bits.extend(struct.pack('>'+precision,x))
    for key in ('state','velocity','drive_targets'):add(data[key])
    for name in BODY_NAMES:
        for key in ('position','orientation','velocity'):add(data['bodies'][name][key])
    for key,precision in (('base_world_pose','d'),('robot_base_world_pose','f')):
        for field in ('position','orientation'):add(data[key][field],precision)
    return hashlib.sha256(bits).hexdigest()


def compare(actual,expected):
    left,right=physical_state(actual),physical_state(expected);differences=[]
    def walk(a,b,path):
        if isinstance(a,dict):
            for key in a:walk(a[key],b[key],path+'.'+key)
        else:
            precision='d' if path.startswith('scene.base_world_pose.') else 'f'
            for i,(x,y) in enumerate(zip(a,b)):
                if struct.pack('>'+precision,x)!=struct.pack('>'+precision,y):
                    differences.append(dict(field=path+f'[{i}]',actual=x,expected=y,
                        actual_bits=struct.pack('>'+precision,x).hex(),expected_bits=struct.pack('>'+precision,y).hex()))
    walk(left,right,'scene')
    return dict(schema=SCHEMA,passed=not differences,actual_sha256=identity(actual),expected_sha256=identity(expected),
        differences=differences,excluded_fields=['sim_ns','physics_time_s','epoch','wall_profile_native'])


def require_identical(actual,expected):
    proof=compare(actual,expected)
    if not proof['passed']:raise ValueError('Exact physical restore differs: '+json.dumps(proof,sort_keys=True))
    return proof


def orientation_geodesic_deg(a,b):
    na=math.sqrt(sum(v*v for v in a));nb=math.sqrt(sum(v*v for v in b))
    if min(na,nb)<1e-10:raise ValueError('Invalid physical quaternion')
    if a==b:return 0.
    w,x,y,z=[v/na for v in a];v,i,j,k=[t/nb for t in b]
    real=w*v+x*i+y*j+z*k
    imaginary=(-w*i+x*v-y*k+z*j,-w*j+y*v+x*k-z*i,-w*k+z*v-x*j+y*i)
    return math.degrees(2*math.atan2(math.sqrt(sum(t*t for t in imaginary)),abs(real)))


def pose_agreement(actual,expected):
    """Lead-approved independent-readback policy; copied observation fields stay exact."""
    left,right=physical_state(actual),physical_state(expected);bits=compare(actual,expected)
    strict=[d for d in bits['differences'] if not any(d['field'].startswith('scene.bodies.'+n+'.'+k+'[')
        for n in BODY_NAMES for k in ('position','orientation'))]
    bodies={}
    for name in BODY_NAMES:
        a,b=left['bodies'][name],right['bodies'][name];delta=[x-y for x,y in zip(a['position'],b['position'])]
        position=math.sqrt(sum(v*v for v in delta));angle=orientation_geodesic_deg(a['orientation'],b['orientation'])
        bodies[name]=dict(position_delta_m=delta,position_error_m=position,orientation_geodesic_deg=angle,
            position_passed=position<=TRIPLET_POSE_POLICY['position_m'],orientation_passed=angle<=TRIPLET_POSE_POLICY['orientation_geodesic_deg'])
    return dict(passed=not strict and all(b['position_passed'] and b['orientation_passed'] for b in bodies.values()),
        policy=copy.deepcopy(TRIPLET_POSE_POLICY),per_body=bodies,strict_differences=strict,exact_bits=bits)


def require_pose_agreement(actual,expected):
    proof=pose_agreement(actual,expected)
    if not proof['passed']:raise ValueError('Triplet physical agreement failed: '+json.dumps(proof,sort_keys=True))
    return proof


def prior_agreements(actual,previous):
    return [dict(previous_member=i,proof=require_pose_agreement(actual,old)) for i,old in enumerate(previous)]
