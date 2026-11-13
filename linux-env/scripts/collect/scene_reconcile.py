"""Authoritative MoveIt reset: world objects, attachments and collision policy."""
import math
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
    pairs.add(tuple(sorted(('base','ground_plane'))))
    pairs.update(tuple(sorted((name,'ground_plane'))) for name in world_ids)
    names = sorted({name for pair in pairs for name in pair})
    return dict(names=names, allowed=[(a==b or tuple(sorted((a,b))) in pairs)
                                    for a in names for b in names])


def policy_matches(acm, policy):
    names=list(acm.entry_names);expected=policy['names']
    if len(names)!=len(set(names)) or set(names)!=set(expected) or len(acm.entry_values)!=len(names):return False
    if acm.default_entry_names or acm.default_entry_values:return False
    if any(len(row.enabled)!=len(names) for row in acm.entry_values):return False
    order={name:i for i,name in enumerate(names)}
    return all(bool(acm.entry_values[order[a]].enabled[order[b]])==value
               for (a,b),value in zip(((a,b) for a in expected for b in expected),policy['allowed']))


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
                        removed_world=[obj.id for obj in world], verified=True, query_components=FULL_SCENE)

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
        return dict(passed=bool(poses['passed'] and feedback and set(world)==expected and not attachments
                                and policy_matches(observed.allowed_collision_matrix,policy) and ground),
                    poses=poses, physical_feedback=feedback, ground=ground, unexpected_world=sorted(set(world)-expected),
                    missing_world=sorted(expected-set(world)), attached=attachments,
                    collision_policy=policy_matches(observed.allowed_collision_matrix,policy),query_components=FULL_SCENE)
