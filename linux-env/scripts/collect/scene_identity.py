"""Exact native physical-state identity, independent of clocks and episode labels."""
import hashlib,json,math,numbers,struct

BODY_NAMES=tuple(sorted([f'{c}_2x3_{i}' for c in ('red','green','blue') for i in range(2)]+
                        ['cup_red','cup_green','cup_blue']))
SCHEMA='soarm101-native-physical-state-v1'


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
