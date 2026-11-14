"""Read-only authored finger-box kinematics, independent of ROS and planning.

The model comes from the held production USD stage. Physics feedback supplies
the base pose and joint positions; this predicts rigid geometry, not forces.
"""
import hashlib,json,math
import numpy as np
from contract import JOINTS


def rotation(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<.1:raise ValueError('Invalid box quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
        [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
        [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])


def transform(position,quaternion):
    if np.asarray(position).shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid rigid position')
    t=np.eye(4);t[:3,:3]=rotation(quaternion);t[:3,3]=position
    return t


def read_model(stage,root='/World/SO_ARM101',effective_finger_offsets=None,effective_block_offsets=None,include_all_links=False):
    """Serialize joint frames and enabled native primitive finger shapes."""
    from pxr import Usd,UsdGeom,UsdPhysics
    def quat(q):return [q.GetReal(),*q.GetImaginary()]
    base=root+'/base_link';joints=[];boxes=[];all_boxes=[];cache=UsdGeom.XformCache()
    parent=stage.GetPrimAtPath(root)
    if not parent.IsValid():raise ValueError('Missing robot root for finger model')
    for p in Usd.PrimRange(parent):
        if p.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(p);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets()
            if not a:continue
            if len(a)!=1 or len(b)!=1:raise ValueError('Incomplete finger-model joint')
            axis=UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get() if p.IsA(UsdPhysics.RevoluteJoint) else None
            if axis and p.GetName() not in JOINTS:raise ValueError('Unknown moving joint in finger model')
            joints.append(dict(name=p.GetName(),body0=str(a[0]),body1=str(b[0]),axis=axis,
                local0=transform(j.GetLocalPos0Attr().Get(),quat(j.GetLocalRot0Attr().Get())).tolist(),
                local1=transform(j.GetLocalPos1Attr().Get(),quat(j.GetLocalRot1Attr().Get())).tolist()))
        if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():continue
        body=p
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body.IsValid():raise ValueError('Robot collider has no rigid body: '+str(p.GetPath()))
        is_finger=body.GetName() in ('gripper_link','moving_jaw_so101_v1_link')
        if not is_finger and not include_all_links:continue
        if not p.IsA(UsdGeom.Cube):raise ValueError('Unexpected non-box finger collision geometry: '+str(p.GetPath()))
        relative,reset=cache.ComputeRelativeTransform(p,body)
        if reset:raise ValueError('Finger collider resets its parent transform')
        relative=np.asarray(relative).T
        lengths=np.linalg.norm(relative[:3,:3],axis=0);R=relative[:3,:3]/lengths
        if not np.isfinite(relative).all() or min(lengths)<=0 or not np.allclose(R.T@R,np.eye(3),atol=1e-9):raise ValueError('Sheared finger box')
        size=float(UsdGeom.Cube(p).GetSizeAttr().Get())
        shape=dict(path=str(p.GetPath()),body=str(body.GetPath()),center_link=relative[:3,3].tolist(),
            rotation_link=R.tolist(),half_extents_m=(lengths*size/2).tolist())
        if is_finger:boxes.append(shape)
        all_boxes.append(shape)
    model=dict(version=1,root_body=base,joints=joints,robot_boxes=boxes)
    if include_all_links:model['all_robot_boxes']=all_boxes
    if effective_finger_offsets is not None:model['effective_finger_offsets']=effective_finger_offsets
    if effective_block_offsets is not None:model['effective_block_offsets']=effective_block_offsets
    FingerModel(model)
    model['sha256']=hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return model


class FingerModel:
    def __init__(self,model,include_all=False):
        if not isinstance(model,dict):raise ValueError('Native finger model is required')
        self.root=model['root_body'];self.joints=[];self.shapes=[]
        if 'sha256' in model:
            identity={k:v for k,v in model.items() if k!='sha256'}
            expected=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if model['sha256']!=expected:raise ValueError('Native finger model identity differs')
        for j in model['joints']:
            if j['axis'] not in (None,'X','Y','Z'):raise ValueError('Unknown finger-model joint axis')
            for field in ('local0','local1'):
                m=np.asarray(j[field])
                if (m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0.,0.,0.,1.],atol=1e-9)
                        or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-6)):raise ValueError('Non-rigid finger-model joint frame')
            self.joints.append(dict(j,inverse1=np.linalg.inv(np.asarray(j['local1'])),matrix0=np.asarray(j['local0'])))
        if len({j['name'] for j in self.joints})!=len(self.joints) or len({j['body1'] for j in self.joints})!=len(self.joints):raise ValueError('Ambiguous finger-model joint tree')
        moving=[j['name'] for j in self.joints if j['axis']]
        if len(moving)!=6 or set(moving)!=set(JOINTS):raise ValueError('Finger model must contain the six named joints')
        shapes=model['all_robot_boxes'] if include_all else model['robot_boxes']
        if not shapes:raise ValueError('Robot collision boxes are required')
        for s in shapes:
            if not include_all and not any('/'+n+'/' in s['path'] for n in ('gripper_link','moving_jaw_so101_v1_link')):continue
            if 'center_link' in s:
                body=s['body'];center=np.asarray(s['center_link']);R=np.asarray(s['rotation_link']);half=np.asarray(s['half_extents_m'])
            else:
                body=s['path'].split('/collisions/')[0];lo=np.asarray(s['low']);hi=np.asarray(s['high']);center=(lo+hi)/2;R=np.eye(3);half=(hi-lo)/2
            if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (center,half)) or min(half)<=0 or R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-9):raise ValueError('Malformed finger box')
            self.shapes.append((s['path'],body,center,R,half))
        if not {'gripper_link','moving_jaw_so101_v1_link'}.issubset({body.rsplit('/',1)[-1] for _,body,*_ in self.shapes}):raise ValueError('Both finger shapes are required')
        connected=self.poses([0.]*6)
        if any(body not in connected for _,body,*_ in self.shapes):raise ValueError('Finger collision body is disconnected')

    def radius_bounds(self):
        """Triangle-inequality bounds from each revolute origin to each box."""
        parent={j['body1']:j for j in self.joints};bounds=dict.fromkeys(JOINTS,0.)
        for _,body,center,R,half in self.shapes:
            chain=[];cursor=body
            while cursor!=self.root:
                joint=parent[cursor];chain.append(joint);cursor=joint['body0']
            for k,j in enumerate(chain):
                if not j['axis']:continue
                distance=float(np.linalg.norm(center-np.asarray(chain[0]['local1'])[:3,3])+np.linalg.norm(half))
                for index in range(k):
                    distance+=float(np.linalg.norm(np.asarray(chain[index]['local0'])[:3,3]-np.asarray(chain[index+1]['local1'])[:3,3]))
                bounds[j['name']]=max(bounds[j['name']],distance)
        return [bounds[n] for n in JOINTS]

    def poses(self,q,base=None):
        if len(q)!=6 or not np.isfinite(q).all():raise ValueError('Invalid finger-model joints')
        out={self.root:np.eye(4) if base is None else transform(base['position'],base['orientation'])};todo=list(self.joints)
        while todo:
            done=[]
            for j in todo:
                if j['body0'] not in out:continue
                r=np.eye(4)
                if j['axis']:
                    i={'X':0,'Y':1,'Z':2}[j['axis']];a=q[JOINTS.index(j['name'])];c,s=math.cos(a),math.sin(a);u=(i+1)%3;v=(i+2)%3
                    r[u,u]=r[v,v]=c;r[u,v]=-s;r[v,u]=s
                out[j['body1']]=out[j['body0']]@j['matrix0']@r@j['inverse1'];done.append(j['name'])
            if not done:raise ValueError('Disconnected finger-model joint tree')
            todo=[j for j in todo if j['name'] not in done]
        return out

    def boxes(self,q,base=None):
        transforms=self.poses(q,base);result=[]
        for path,body,center,R,half in self.shapes:
            t=transforms[body];world=t[:3,:3]@center+t[:3,3];axes=t[:3,:3]@R
            result.append(dict(path=path,rotation=axes,center_world=world,half_extents_m=half,
                min_z_m=float(world[2]-abs(axes[2])@half)))
        return result


def finger_boxes(model,q,base=None):return FingerModel(model).boxes(q,base)


def separation(a,b):
    """SAT lower bound on separation (negative when boxes intersect)."""
    A=np.asarray(a['rotation']);B=np.asarray(b['rotation']);ha=np.asarray(a['half_extents_m']);hb=np.asarray(b['half_extents_m']);delta=np.asarray(b['center_world'])-np.asarray(a['center_world'])
    axes=np.concatenate((A.T,B.T,np.cross(A.T[:,None,:],B.T[None,:,:]).reshape(-1,3)))
    norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-10]/norms[norms>1e-10,None]
    return float(np.max(abs(axes@delta)-abs(axes@A)@ha-abs(axes@B)@hb))


def body_box(body,corners):
    local=np.asarray(corners,dtype=float);R=rotation(body['orientation'])
    if local.shape!=(8,3) or not np.isfinite(local).all():raise ValueError('Native block corners required')
    if np.asarray(body['position']).shape!=(3,) or not np.isfinite(body['position']).all():raise ValueError('Invalid block position')
    lo=local.min(axis=0);hi=local.max(axis=0)
    return dict(rotation=R,center_world=np.asarray(body['position'])+R@((lo+hi)/2),half_extents_m=(hi-lo)/2)
