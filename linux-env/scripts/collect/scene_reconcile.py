"""Authoritative MoveIt reset: world objects, attachments and collision policy."""
import math
import hashlib
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from scene_sync import collision_pose_errors, physical_pose_caches, multiply
from success import rotate

FULL_SCENE = 1023


def ros_messages():
    from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject, AllowedCollisionEntry
    from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
    return SimpleNamespace(Scene=PlanningScene, Object=CollisionObject, Attached=AttachedCollisionObject,
                           Entry=AllowedCollisionEntry, Get=GetPlanningScene, Apply=ApplyPlanningScene)


def collision_policy(srdf, world_ids=()):
    pairs = {tuple(sorted((n.attrib['link1'],n.attrib['link2'])))
             for n in ET.fromstring(srdf).findall('disable_collisions')}
    world_ids=set(world_ids)
    # Empty-world reset has only SRDF robot pairs. MoveIt removes an object's
    # ACM entries on world destruction (including when pushing scene diffs).
    # Ground exemptions belong to the rebuilt scene that contains the ground.
    if 'ground_plane' in world_ids:
        pairs.add(tuple(sorted(('base','ground_plane'))))
        pairs.update(tuple(sorted((name,'ground_plane'))) for name in world_ids)
    names = sorted({name for pair in pairs for name in pair})
    return dict(names=names, allowed=[(a!=b and tuple(sorted((a,b))) in pairs)
                                    for a in names for b in names])


def policy_matches(acm, policy):
    names=list(acm.entry_names);expected=policy['names']
    if len(names)!=len(set(names)) or len(acm.entry_values)!=len(names):return False
    if acm.default_entry_names or acm.default_entry_values:return False
    if any(len(row.enabled)!=len(names) for row in acm.entry_values):return False
    order={name:i for i,name in enumerate(names)}
    wanted={(a,b):value for (a,b),value in
            zip(((a,b) for a in expected for b in expected),policy['allowed'])}
    # MoveIt imports only i<j and omits names without explicit/default entries.
    # Missing pairs therefore mean false; self-pairs have no collision meaning.
    union=set(names)|set(expected)
    return all((bool(acm.entry_values[order[a]].enabled[order[b]])
                if a in order and b in order else False)==wanted.get((a,b),False)
               for a in union for b in union if a!=b)


def native_scene_summary(scene):
    """Retain the service representation, including diagonals, for live proof."""
    acm=scene.allowed_collision_matrix
    return dict(query_components=FULL_SCENE,
        world_ids=sorted(obj.id for obj in scene.world.collision_objects),
        attached_ids=sorted(obj.object.id for obj in scene.robot_state.attached_collision_objects),
        octomap_bytes=len(scene.world.octomap.octomap.data),
        robot_joints=dict(names=list(scene.robot_state.joint_state.name),
                          positions=list(scene.robot_state.joint_state.position)),
        acm=dict(entry_names=list(acm.entry_names),entry_values=[list(row.enabled) for row in acm.entry_values],
                 default_entry_names=list(acm.default_entry_names),default_entry_values=list(acm.default_entry_values)))


def shape_content(kind, shape):
    """Mesh identity independent of vertex/triangle ordering and winding."""
    if kind!='mesh':
        return dict(kind='primitive', type=int(shape.type), dimensions=list(shape.dimensions))
    vertices=[tuple(float(getattr(v,k)) for k in ('x','y','z')) for v in shape.vertices]
    triangles=[]
    for triangle in shape.triangles:
        indices=list(triangle.vertex_indices)
        if len(indices)!=3 or any(i<0 or i>=len(vertices) for i in indices):
            raise ValueError('Invalid mesh triangle indices')
        triangles.append(sorted(vertices[i] for i in indices))
    return dict(kind='mesh', vertices=sorted(vertices), triangles=sorted(triangles))


def geometry_check(obj, kind, shape):
    def close(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
        if isinstance(a,(int,float)):
            return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-7
        return a==b
    try:
        if obj is None or getattr(obj,'planes',[]):raise ValueError('Missing object or unexpected plane')
        shapes=list(obj.meshes)+list(obj.primitives)
        if len(shapes)!=1 or bool(obj.meshes)!=(kind=='mesh'):raise ValueError('Shape kind/count mismatch')
        expected=shape_content(kind,shape);actual=shape_content(kind,shapes[0])
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return dict(passed=close(expected,actual),kind=expected['kind'],expected_sha256=digest(expected),
                    observed_sha256=digest(actual))
    except (ValueError,TypeError,AttributeError,IndexError) as exc:
        return dict(passed=False,error=str(exc))


def ground_matches(obj, snapshot, height):
    if obj is None or len(obj.primitives)!=1 or obj.meshes or len(obj.primitive_poses)!=1:return False
    primitive=obj.primitives[0]
    if primitive.type!=1 or len(primitive.dimensions)!=3:return False
    if max(abs(a-b) for a,b in zip(primitive.dimensions,[2.,2.,.01]))>1e-6:return False
    pose=obj.primitive_poses[0]
    pos=[getattr(pose.position,k) for k in ('x','y','z')]
    quat=[getattr(pose.orientation,k) for k in ('w','x','y','z')]
    parent=getattr(obj,'pose',None)
    if parent is not None:
        pq=[getattr(parent.orientation,k) for k in ('w','x','y','z')]
        if sum(v*v for v in pq)<1e-15:pq=[1,0,0,0]
        pos=[a+getattr(parent.position,k) for a,k in zip(rotate(pq,pos),('x','y','z'))]
        quat=multiply(pq,quat)
    if obj.header.frame_id=='world':
        base=snapshot['base_world_pose'];inv=[base['orientation'][0]]+[-v for v in base['orientation'][1:]]
        pos=rotate(inv,[a-b for a,b in zip(pos,base['position'])]);quat=multiply(inv,quat)
    elif obj.header.frame_id!='base':return False
    norm=math.sqrt(sum(v*v for v in quat))
    return (math.isfinite(sum(pos)+norm) and norm>1e-10 and
            max(abs(a-b) for a,b in zip(pos,[0.,0.,height-.005]))<=1e-5 and
            max(abs(v/norm) for v in quat[1:])<=1e-5)


class SceneReconciler:
    def __init__(self, node, messages, srdf):
        self.node,self.msg,self.srdf=node,messages,srdf

    def query(self):
        req=self.msg.Get.Request();req.components.components=FULL_SCENE
        response=self.node._wait_future(self.node._get_scene_client.call_async(req),timeout_sec=3.)
        if response is None:raise TimeoutError('Full planning scene query did not complete')
        return response.scene

    def apply(self, scene):
        req=self.msg.Apply.Request();req.scene=scene
        # This is the shared checked mutation client; unresolved completion is
        # terminal even if a caller catches the raised error.
        return self.node._wait_future(self.node._apply_scene_client.call_async(req),timeout_sec=5.)

    def diff(self):
        scene=self.msg.Scene();scene.is_diff=True;scene.robot_state.is_diff=True
        return scene

    def set_policy(self, world_ids=()):
        policy=collision_policy(self.srdf,world_ids)
        scene=self.diff();acm=scene.allowed_collision_matrix
        acm.entry_names=policy['names']
        size=len(acm.entry_names)
        for start in range(0,size*size,size):
            row=self.msg.Entry();row.enabled=policy['allowed'][start:start+size];acm.entry_values.append(row)
        response=self.apply(scene)
        if not response.success:raise RuntimeError('Collision policy replacement was rejected')
        return policy

    def clear(self):
        """Clear remote state regardless of local attached-name/tracker fields."""
        with self.node._motion_guard():
            self.node._apply_scene_client.require_idle()
            initial=self.query()
            attached=list(initial.robot_state.attached_collision_objects)
            if attached:
                scene=self.diff()
                for old in attached:
                    obj=self.msg.Attached();obj.link_name=old.link_name
                    obj.object.id=old.object.id;obj.object.operation=self.msg.Object.REMOVE
                    scene.robot_state.attached_collision_objects.append(obj)
                self.apply(scene)
            observed=self.query() if attached else initial
            if observed.robot_state.attached_collision_objects:
                raise RuntimeError('Planning scene still contains attached objects after reset removal')
            # Detachment can create world objects, so query after detaching.
            world=list(observed.world.collision_objects)
            if world:
                scene=self.diff()
                for old in world:
                    obj=self.msg.Object();obj.id=old.id;obj.operation=self.msg.Object.REMOVE
                    scene.world.collision_objects.append(obj)
                self.apply(scene)
            policy=self.set_policy()
            final=self.query()
            octomap=getattr(getattr(final.world,'octomap',None),'octomap',None)
            if octomap is not None and octomap.data:
                raise RuntimeError('Unexpected occupancy map; fresh control stack required')
            if final.world.collision_objects or final.robot_state.attached_collision_objects or not policy_matches(final.allowed_collision_matrix,policy):
                raise RuntimeError('Planning scene reset postcondition did not match remote state')
            self.node._attached_lego_name=None;self.node._attached_lego_tcp_offset=None
            self.node._lego_collision_names=[];self.node._cup_collision_names=[]
            return dict(removed_attached=[obj.object.id for obj in attached],
                        removed_world=[obj.id for obj in world], verified=True, query_components=FULL_SCENE,
                        native_scene=native_scene_summary(final))

    def verify(self, snapshot, joint_names):
        observed=self.query()
        objects,cups=physical_pose_caches(snapshot)
        expected={'ground_plane'}|{'lego_'+name for name in objects}|{'cup_'+name for name in cups}
        world={obj.id:obj for obj in observed.world.collision_objects}
        state=dict(zip(observed.robot_state.joint_state.name,observed.robot_state.joint_state.position))
        feedback=all(name in state and abs(state[name]-q)<1e-5 for name,q in zip(joint_names,snapshot['state']))
        policy=collision_policy(self.srdf,expected)
        poses=collision_pose_errors(observed.world.collision_objects,snapshot)
        attachments=[obj.object.id for obj in observed.robot_state.attached_collision_objects]
        ground=ground_matches(world.get('ground_plane'),snapshot,self.node._ground_z_var.get())
        geometry={}
        for name in objects:
            kind,shape=self.node._build_lego_geometry(name)
            geometry['lego_'+name]=geometry_check(world.get('lego_'+name),kind,shape)
        cup_kind,cup_shape=self.node._build_cup_geometry()
        for name in cups:
            geometry['cup_'+name]=geometry_check(world.get('cup_'+name),cup_kind,cup_shape)
        geometry_ok=all(check['passed'] for check in geometry.values())
        native=native_scene_summary(observed)
        occupancy_empty=native['octomap_bytes']==0
        return dict(passed=bool(poses['passed'] and feedback and set(world)==expected and not attachments
                                and policy_matches(observed.allowed_collision_matrix,policy) and ground
                                and geometry_ok and occupancy_empty),
                    poses=poses, physical_feedback=feedback, ground=ground, unexpected_world=sorted(set(world)-expected),
                    missing_world=sorted(expected-set(world)), attached=attachments,
                    geometry=geometry, occupancy_empty=occupancy_empty,native_scene=native,
                    collision_policy=policy_matches(observed.allowed_collision_matrix,policy),query_components=FULL_SCENE)
