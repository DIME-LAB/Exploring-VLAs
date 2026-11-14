"""USD-joint forward kinematics and oriented primitive-box collision checks."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
import shared
from command_geometry import final_command
from so_arm101_control.grasp_geometry import JOINT_LIMITS
from contract import JOINTS


def pose_matrix(position, quaternion):
    w,x,y,z=np.asarray(quaternion,dtype=float)/np.linalg.norm(quaternion)
    m=np.eye(4)
    m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    m[:3,3]=position
    if not np.isfinite(m).all():raise ValueError('Non-finite pose')
    return m


def rotation(axis, angle):
    v=np.eye(3)['XYZ'.index(axis)]
    cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    result=np.eye(4)
    result[:3,:3]=np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)
    return result


def collider_in_body(parent_world,collider_world,size):
    """Physics poses are rigid; retain all USD inherited scale in metre boxes."""
    parent_rigid=np.asarray(parent_world,float).copy()
    parent_rigid[:3,:3]/=np.linalg.norm(parent_rigid[:3,:3],axis=0)
    matrix=np.linalg.inv(parent_rigid)@np.asarray(collider_world,float)
    scale=np.linalg.norm(matrix[:3,:3],axis=0)
    matrix[:3,:3]/=scale
    if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-7):
        raise RuntimeError('Sheared native box cannot be represented as an OBB')
    return matrix,scale*float(size)/2


@dataclass
class Box:
    name: str
    body: str
    matrix: np.ndarray
    half: np.ndarray

    def moved(self, matrix):
        return Box(self.name,self.body,matrix@self.matrix,self.half)

    @property
    def low(self):return self.matrix[:3,3]-abs(self.matrix[:3,:3])@self.half


def overlaps(a,b,padding=.0001):
    """15-axis SAT; parallel cross axes are redundant, not separating."""
    axes_a,axes_b=a.matrix[:3,:3],b.matrix[:3,:3]
    ha,hb=a.half+padding,b.half+padding
    delta=b.matrix[:3,3]-a.matrix[:3,3]
    axes=[*axes_a.T,*axes_b.T]
    axes.extend(np.cross(x,y) for x in axes_a.T for y in axes_b.T)
    for axis in axes:
        if np.linalg.norm(axis)<1e-10:continue
        if abs(delta@axis)>ha@abs(axes_a.T@axis)+hb@abs(axes_b.T@axis):return False
    return True


class Geometry:
    def __init__(self,spec):
        self.spec=spec
        self.joints=spec['joints']
        self.frames={j['name']:(np.asarray(j['frame0']),np.linalg.inv(np.asarray(j['frame1'])))
            for j in self.joints}
        self.boxes=[Box(s['name'],s['body'],np.asarray(s['matrix'],float),
                        np.asarray(s['half'],float)) for s in spec['boxes']]
        if not self.boxes or not self.joints:raise ValueError('Missing native primitive geometry')
        self.allowed={frozenset(p) for p in spec['allowed_pairs']}
        self.selected=None;self.contact=False;self.carried=None
        self.snapshot=None;self.roots={name:np.asarray(matrix,float) for name,matrix in spec['roots'].items()}
        self.root_motion={}

    def fk(self,q,roots=None):
        links=dict(self.roots if roots is None else roots)
        pending=list(self.joints)
        values=dict(zip(JOINTS,q))
        while pending:
            progressed=False
            for joint in list(pending):
                if joint['parent'] not in links:continue
                name=joint['name']
                if joint['type']=='revolute' and name not in values:
                    raise RuntimeError('Unmapped revolute joint: '+name)
                turn=rotation(joint['axis'],values[name]) if joint['type']=='revolute' else np.eye(4)
                frame0,inverse1=self.frames[name]
                links[joint['child']]=links[joint['parent']]@frame0@turn@inverse1
                pending.remove(joint);progressed=True
            if not progressed:raise RuntimeError('Disconnected native joint tree')
        return links

    def update(self,snapshot):
        roots={name:np.asarray(snapshot['links'][name],float) for name in self.spec['roots']}
        motion={}
        for name,matrix in roots.items():
            if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3],[0,0,0,1],rtol=0,atol=1e-7) or
                not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),rtol=0,atol=1e-6) or
                not np.isclose(np.linalg.det(matrix[:3,:3]),1.,rtol=0,atol=1e-6)):
                raise RuntimeError('Invalid measured articulation root: '+name)
            delta=np.linalg.inv(np.asarray(self.spec['roots'][name],float))@matrix
            motion[name]=dict(translation_m=float(np.linalg.norm(delta[:3,3])),
                rotation_rad=float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
        predicted=self.fk(snapshot['state'],roots)
        for name,actual in snapshot['links'].items():
            if name not in predicted:raise RuntimeError('Unmapped physical link: '+name)
            if not np.allclose(predicted[name],actual,rtol=0,atol=.0005):
                raise RuntimeError('USD FK differs from measured physical link: '+name)
        self.roots={name:matrix.copy() for name,matrix in roots.items()}
        self.root_motion=motion;self.snapshot=snapshot

    def bind_carried(self,snapshot):
        link=self.spec['gripper_body']
        body=snapshot['bodies'][self.selected]
        self.carried=np.linalg.inv(np.asarray(snapshot['links'][link]))@pose_matrix(body['position'],body['orientation'])

    def check(self,q):
        if self.snapshot is None:raise RuntimeError('No current physical geometry')
        links=self.fk(q);robot=[];obstacles=[]
        for box in self.boxes:
            if box.body in links:robot.append(box.moved(links[box.body]))
            elif box.body in self.snapshot['bodies']:
                body=self.snapshot['bodies'][box.body]
                if box.body==self.selected and self.carried is not None:
                    robot.append(box.moved(links[self.spec['gripper_body']]@self.carried))
                else:obstacles.append(box.moved(pose_matrix(body['position'],body['orientation'])))
            else:raise RuntimeError('Collider has no physical body: '+box.body)
        all_boxes=[*robot,*obstacles]
        centers=np.asarray([b.matrix[:3,3] for b in all_boxes])
        extents=np.asarray([abs(b.matrix[:3,:3])@b.half+.0002 for b in all_boxes])
        candidates=np.all(abs(centers[:len(robot),None]-centers[None])<=
            extents[:len(robot),None]+extents[None],axis=2)
        for i,a in enumerate(robot):
            if a.body not in self.spec['ground_allowed'] and a.low[2]<self.spec['ground_z']-.0001:
                return False,[SimpleNamespace(contact_body_1=a.name,contact_body_2='ground_plane',depth=-a.low[2])]
            for j in np.flatnonzero(candidates[i]):
                if j<=i:continue
                b=all_boxes[j]
                if a.body==b.body or frozenset((a.body,b.body)) in self.allowed:continue
                if self.selected in (a.body,b.body) and self.contact:
                    other=a if b.body==self.selected else b
                    if other.name in self.spec['finger_pad_shapes']:continue
                if overlaps(a,b):
                    return False,[SimpleNamespace(contact_body_1=a.name,contact_body_2=b.name,depth=0.)]
        return True,[]

    def __call__(self,candidate,snapshot):
        self.update(snapshot)
        result=final_command(candidate,snapshot['state'],snapshot['drive_targets'],
            self.check,[JOINT_LIMITS[n] for n in JOINTS])
        result['method']='native USD primitive boxes; 15-axis SAT; sampled joint sweeps'
        return result
