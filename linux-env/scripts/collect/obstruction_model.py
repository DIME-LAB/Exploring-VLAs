"""Offline test model from retained authored joint frames and primitive boxes.

It reconstructs geometry; it does not measure solver link deflection or forces.
"""
import itertools,math
import numpy as np
from contract import JOINTS


def poses(fixture,q):
    out={'/so101_new_calib/base_link':np.eye(4)};todo=list(fixture['joints'])
    while todo:
        done=[]
        for j in todo:
            if j['body0'] not in out:continue
            r=np.eye(4)
            if j['axis']:
                i={'X':0,'Y':1,'Z':2}[j['axis']];a=q[JOINTS.index(j['name'])];c,s=math.cos(a),math.sin(a);u=(i+1)%3;v=(i+2)%3
                r[u,u]=r[v,v]=c;r[u,v]=-s;r[v,u]=s
            out[j['body1']]=out[j['body0']]@np.asarray(j['local0'])@r@np.linalg.inv(j['local1']);done.append(j['name'])
        if not done:raise ValueError('Disconnected retained joint tree')
        todo=[j for j in todo if j['name'] not in done]
    return out


def finger_boxes(fixture,q):
    transforms=poses(fixture,q);boxes=[]
    for shape in fixture['robot_boxes']:
        if not any('/'+name+'/' in shape['path'] for name in ('gripper_link','moving_jaw_so101_v1_link')):continue
        t=transforms[shape['path'].split('/collisions/')[0]];lo=np.asarray(shape['low']);hi=np.asarray(shape['high']);c=(lo+hi)/2
        pts=np.array([list(p)+[1.] for p in itertools.product(*zip(lo,hi))]);world=(t@pts.T).T[:,:3]
        boxes.append(dict(path=shape['path'],rotation=t[:3,:3],center_world=(t@np.r_[c,1.])[:3],half_extents_m=(hi-lo)/2,min_z_m=float(world[:,2].min())))
    return boxes


def separation(a,b):
    A=np.asarray(a['rotation']);B=np.asarray(b['rotation']);ha=np.asarray(a['half_extents_m']);hb=np.asarray(b['half_extents_m']);delta=np.asarray(b['center_world'])-np.asarray(a['center_world']);gaps=[]
    for n in [*A.T,*B.T,*[np.cross(x,y) for x in A.T for y in B.T]]:
        size=np.linalg.norm(n)
        if size<1e-10:continue
        n=n/size;gaps.append(abs(float(delta@n))-float(ha@abs(A.T@n)+hb@abs(B.T@n)))
    return max(gaps)
