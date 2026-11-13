"""Physical-body to MoveIt pose contract, independent of ROS transport."""
import math
from success import rotate

CUP_HEIGHT_M = .0965
DROP_COLORS = {'drop_1':'blue', 'drop_2':'green', 'drop_3':'red'}


def multiply(a, b):
    w,x,y,z = a; v,i,j,k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def physical_pose_caches(snapshot):
    base = snapshot['base_world_pose']
    inv = [base['orientation'][0]] + [-v for v in base['orientation'][1:]]
    objects, cups = {}, {}
    for name, body in snapshot['bodies'].items():
        q = multiply(inv, body['orientation'])
        pos = rotate(inv, [v-o for v,o in zip(body['position'], base['position'])])
        if name.startswith('cup_'):
            offset = rotate(q, [0,0,CUP_HEIGHT_M/2])
            pos = [a+b for a,b in zip(pos,offset)]
            target = next(k for k,v in DROP_COLORS.items() if name == 'cup_'+v)
            dest = cups
        else:
            target = name;dest = objects
        dest[target] = dict(zip(('x','y','z','qw','qx','qy','qz'), pos+q))
    return objects, cups


def collision_pose_errors(collision_objects, snapshot):
    """Compare every reset body, including geometry and parent-frame offsets."""
    objects, cups = physical_pose_caches(snapshot)
    expected = {'lego_'+k:(v, False) for k,v in objects.items()}
    expected.update({'cup_'+k:(v, True) for k,v in cups.items()})
    actual = {co.id:co for co in collision_objects}
    errors = {}
    for name, (target, cup) in expected.items():
        if name not in actual:
            errors[name] = dict(error='missing collision object');continue
        co = actual[name]
        if co.header.frame_id not in ('base', 'world', '/World/base'):
            errors[name] = dict(error='unsupported planning frame '+co.header.frame_id);continue
        mesh = bool(co.mesh_poses)
        poses = list(co.mesh_poses)+list(co.primitive_poses)
        if len(poses) != 1:
            errors[name] = dict(error='exactly one collision geometry pose required');continue
        pose = poses[0]
        pos = [getattr(pose.position,k) for k in ('x','y','z')]
        q = [getattr(pose.orientation,k) for k in ('w','x','y','z')]
        parent = getattr(co,'pose',None)
        if parent is not None:
            pq = [getattr(parent.orientation,k) for k in ('w','x','y','z')]
            if sum(v*v for v in pq) < 1e-15:
                pq = [1,0,0,0]  # Unspecified optional object pose in ROS messages.
            pos = [a+getattr(parent.position,k) for a,k in zip(rotate(pq,pos),('x','y','z'))]
            q = multiply(pq,q)
        if co.header.frame_id == 'world':
            base = snapshot['base_world_pose']
            inv = [base['orientation'][0]]+[-v for v in base['orientation'][1:]]
            pos = rotate(inv,[v-o for v,o in zip(pos,base['position'])]);q=multiply(inv,q)
        tq = [target[k] for k in ('qw','qx','qy','qz')]
        tp = [target[k] for k in ('x','y','z')]
        if cup and mesh:
            offset=rotate(tq,[0,0,CUP_HEIGHT_M/2]);tp=[a-b for a,b in zip(tp,offset)]
        norm=math.sqrt(sum(v*v for v in q))
        if norm < 1e-10 or not math.isfinite(norm):
            errors[name] = dict(error='invalid collision orientation');continue
        q = [v/norm for v in q]
        position_error = max(abs(a-b) for a,b in zip(pos,tp))
        orientation_error = min(max(abs(a-b) for a,b in zip(q,tq)),max(abs(a+b) for a,b in zip(q,tq)))
        errors[name] = dict(position_error_m=position_error, quaternion_error=orientation_error,
                            passed=math.isfinite(position_error+orientation_error) and position_error<=1e-5 and orientation_error<=1e-5)
    return dict(passed=bool(errors) and all(v.get('passed',False) for v in errors.values()), objects=errors)
