"""Scene-request completion, authoritative reset and diagnostic regressions."""
import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch
import threading
import unittest
import numpy as np
from protocol import OwnershipStalled
from scene_reconcile import SceneReconciler, collision_policy, policy_matches, FULL_SCENE
from storage import SceneSafetyMarker, scene_stack_ready
from test_safety import method
import test_motion_operations as motions
SHARED=motions.SHARED
import test_worker_retirement as workers
from test_review_regressions import load_class, JOINTS
from so_arm101_control.scene_transactions import SceneMutationClient, SceneMutationIncomplete

ROOT=Path(__file__).resolve().parents[3]
SRDF=(ROOT/'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf').read_text()


def pose():return NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=0.))


class Object:
    ADD=0;REMOVE=1
    def __init__(self):
        self.id='';self.operation=0;self.header=NS(frame_id='base')
        self.meshes=[];self.mesh_poses=[];self.primitives=[];self.primitive_poses=[]


class Attached:
    def __init__(self):self.object=Object();self.link_name='';self.touch_links=[]


class Scene:
    def __init__(self):
        self.is_diff=False
        self.world=NS(collision_objects=[],octomap=NS(octomap=NS(data=[])))
        self.robot_state=NS(is_diff=False,joint_state=NS(name=list(JOINTS),position=[0.]*6),attached_collision_objects=[])
        self.allowed_collision_matrix=NS(entry_names=[],entry_values=[],default_entry_names=[],default_entry_values=[])


class Request:
    def __init__(self,**kw):self.components=NS(components=0);self.__dict__.update(kw)


class Primitive:
    BOX=1;CYLINDER=3
    def __init__(self):self.type=0;self.dimensions=[]


class Future:
    def __init__(self,value=None,completed=True):self.value=value;self.completed=completed
    def done(self):return self.completed
    def result(self):return self.value


MESSAGES=NS(Scene=Scene,Object=Object,Attached=Attached,Entry=lambda:NS(enabled=[]),
            Get=NS(Request=Request),Apply=NS(Request=Request))


class SceneServer:
    def __init__(self):
        self.scene=Scene();self.pending_kind=None;self.false_kind=None;self.pending=[];self.queries=[];self.calls=[]
    def kind(self,scene):
        attached=scene.robot_state.attached_collision_objects
        if attached:return 'detach' if attached[0].object.operation==Object.REMOVE else 'attach'
        if scene.world.collision_objects:return 'world_remove' if scene.world.collision_objects[0].operation==Object.REMOVE else 'world_add'
        return 'acm'
    def apply_now(self,scene):
        world={obj.id:obj for obj in self.scene.world.collision_objects}
        attached={obj.object.id:obj for obj in self.scene.robot_state.attached_collision_objects}
        for obj in scene.robot_state.attached_collision_objects:
            if obj.object.operation==Object.REMOVE:
                old=attached.pop(obj.object.id,None)
                if old is not None:world[old.object.id]=old.object
            else:
                attached[obj.object.id]=copy.deepcopy(obj);world.pop(obj.object.id,None)
        for obj in scene.world.collision_objects:
            if obj.operation==Object.REMOVE:world.pop(obj.id,None)
            else:world[obj.id]=copy.deepcopy(obj)
        if scene.allowed_collision_matrix.entry_names:
            self.scene.allowed_collision_matrix=copy.deepcopy(scene.allowed_collision_matrix)
        self.scene.world.collision_objects=list(world.values())
        self.scene.robot_state.attached_collision_objects=list(attached.values())
    def apply(self,request):
        scene=copy.deepcopy(request.scene);kind=self.kind(scene);self.calls.append(kind)
        if kind==self.pending_kind:
            future=Future(completed=False)
            def finish():
                self.apply_now(scene);future.value=NS(success=True);future.completed=True
            self.pending.append(finish);return future
        self.apply_now(scene)
        return Future(NS(success=kind!=self.false_kind))
    def query(self,request):
        self.queries.append(request.components.components)
        return Future(NS(scene=copy.deepcopy(self.scene)))


class SceneLifetimeTests(unittest.TestCase):
    def setup_scene(self):
        c,r,entry,_,_=motions.MotionOperationTests().setup_motion();n=c.node;server=SceneServer();markers=[]
        c.scene_marker=NS(record=lambda error,**kw:markers.append((str(error),kw)))
        n._scene_failure_handler=c.scene_incomplete
        n.objects_lock=threading.RLock();n._drop_lock=threading.RLock()
        n.objects_data={'red_2x3_0':dict(x=.2,y=0.,z=.012,qw=1.)};n._drop_data={}
        n._attached_lego_name=None;n._attached_lego_tcp_offset=None
        n._lego_collision_names=['lego_red_2x3_0'];n._cup_collision_names=[]
        n._ground_z_var=NS(get=lambda:0.);n._ground_plane_var=NS(get=lambda:True)
        n._lookup_bbox=lambda name:dict(sx=.031,sy=.0206,sz=.0128)
        n._wait_arm_at_rest=lambda **kw:True
        n._tf_buffer=NS(lookup_transform=lambda *a:NS(transform=NS(translation=NS(x=.2,y=0.,z=.012),rotation=NS(x=0.,y=0.,z=0.,w=1.))))
        def geometry(name):
            primitive=Primitive();primitive.type=Primitive.BOX;primitive.dimensions=[.031,.0206,.0128]
            return 'box',primitive
        n._build_lego_geometry=geometry
        globals_=dict(MOVEIT_AVAILABLE=True,PlanningSceneMsg=Scene,CollisionObject=Object,AttachedCollisionObject=Attached,
            ApplyPlanningScene=MESSAGES.Apply,GetPlanningSceneSrv=MESSAGES.Get,Pose=pose,SolidPrimitive=Primitive,
            AllowedCollisionEntry=MESSAGES.Entry,_CUP_COLLISION_PADDING=1.,CUP_BODY_HEIGHT_M=.0965,
            _load_cup_mesh=lambda:(None,False),rclpy=NS(time=NS(Time=lambda:None)))
        for name in ('_ensure_lego_state','_attach_lego_to_gripper','_detach_lego_sync','_pose7_to_mat','_mat_to_pose7',
                     '_dispatch_scene_update','_remove_single_lego_from_world','_remove_lego_collision_objects',
                     '_remove_cup_collision_objects','_add_cup_collision_objects','_add_lego_collision_objects',
                     '_acm_allow_ground_rest','_cmd_toggle_ground_plane','_scene_unresolved'):
            fn=method(SHARED,'PickPlaceNode',name);fn=getattr(fn,'__func__',fn);fn.__globals__.update(globals_)
            setattr(n,name,fn if name in ('_pose7_to_mat','_mat_to_pose7') else MethodType(fn,n))
        raw=NS(call_async=server.apply,wait_for_service=lambda **kw:True)
        n._apply_scene_client=SceneMutationClient(raw,n._motion_guard,n._scene_unresolved)
        n._get_scene_client=NS(call_async=server.query,wait_for_service=lambda **kw:True)
        waiter=method(SHARED,'PickPlaceNode','_wait_future');waiter=getattr(waiter,'__func__',waiter);clock=[0.]
        waiter.__globals__['time']=NS(monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
        n._wait_future=waiter
        c.update_pose_cache=MethodType(c.__class__.update_pose_cache,c)
        c.update_pose_cache.__func__.__globals__['physical_pose_caches']=__import__('scene_sync').physical_pose_caches
        c.scene_reset=SceneReconciler(n,MESSAGES,SRDF)
        c.scene_reset.set_policy()
        return c,r,entry,server,markers

    def fixture(self):
        from test_safety import SafetyTests
        snapshot=SafetyTests().snapshot();snapshot.update(state=[0.]*6,base_world_pose=dict(position=[0.]*3,orientation=[1.,0.,0.,0.]))
        snapshot['bodies']['red_2x3_0']['position'][0]=.2
        for color,x in [('green',.35),('blue',.5)]:
            snapshot['bodies']['cup_'+color]=dict(position=[x,0.,0.],orientation=[1.,0.,0.,0.],velocity=[0.]*6)
        return snapshot

    def populate(self,c):
        c.update_pose_cache.__func__.__globals__['physical_pose_caches']=__import__('scene_sync').physical_pose_caches
        c.episode_start_physical=self.fixture()
        c.update_pose_cache(c.episode_start_physical)
        c.sync_scene()

    def test_pending_mutations_are_terminal_even_when_the_server_completes_later(self):
        for kind in ('attach','world_add','world_remove','acm','detach'):
            with self.subTest(kind=kind):
                c,r,_,server,markers=self.setup_scene();n=c.node;server.pending_kind=kind
                if kind=='attach':operation=lambda:n._attach_lego_to_gripper('red_2x3_0')
                elif kind=='world_add':operation=n._add_lego_collision_objects
                elif kind=='world_remove':operation=lambda:n._remove_single_lego_from_world('red_2x3_0')
                elif kind=='acm':operation=lambda:n._acm_allow_ground_rest(['new_block'])
                else:
                    obj=Attached();obj.object.id='lego_red_2x3_0';obj.link_name='tcp_link'
                    server.scene.robot_state.attached_collision_objects=[obj]
                    n._attached_lego_name='red_2x3_0';operation=n._detach_lego_sync
                with self.assertRaises(SceneMutationIncomplete):operation()
                self.assertTrue(c.stop.is_set());self.assertTrue(n._qs_abort_evt.is_set())
                self.assertIsNone(c.generation.epoch);self.assertTrue(c.generation.stalled)
                self.assertEqual(c.events,['pause']);self.assertEqual(len(markers),1)
                self.assertEqual(len(n._apply_scene_client.pending),1)
                self.assertEqual(len(server.pending),1)
                server.pending.pop()()  # Server completion is still possible after a local timeout.
                n._attached_lego_name=None
                previous_calls=list(server.calls)
                with self.assertRaises(SceneMutationIncomplete):c.scene_reset.clear()
                with self.assertRaises(OwnershipStalled):c.timeline(False)
                self.assertEqual(server.calls,previous_calls)

    def test_reset_removes_remote_attachments_world_and_acm_without_local_trackers(self):
        c,r,_,server,_=self.setup_scene();n=c.node
        obj=Attached();obj.object.id='lego_red_2x3_0';obj.link_name='tcp_link'
        server.scene.robot_state.attached_collision_objects=[obj]
        extra=Object();extra.id='old_part';server.scene.world.collision_objects=[extra]
        matrix=server.scene.allowed_collision_matrix
        matrix.default_entry_names=['old_part'];matrix.default_entry_values=[True]
        matrix.entry_values[0].enabled=[True]*len(matrix.entry_names)
        self.assertIsNone(n._attached_lego_name)
        cleared=c.scene_reset.clear()
        self.assertIn('lego_red_2x3_0',cleared['removed_attached'])
        self.assertEqual(set(cleared['removed_world']),{'old_part','lego_red_2x3_0'})
        self.assertTrue(policy_matches(server.scene.allowed_collision_matrix,collision_policy(SRDF)))
        self.assertTrue(all(mask==FULL_SCENE for mask in server.queries[-3:]))
        self.populate(c)
        self.assertTrue(c.scene_pose_check['passed'])
        self.assertFalse(server.scene.robot_state.attached_collision_objects)
        self.assertEqual({obj.id for obj in server.scene.world.collision_objects},
                         {'ground_plane','lego_red_2x3_0','cup_drop_1','cup_drop_2','cup_drop_3'})

    def test_verification_rejects_extra_world_attachment_policy_and_wrong_ground(self):
        c,_,_,server,_=self.setup_scene();self.populate(c);baseline=copy.deepcopy(server.scene)
        for kind in ('world','attached','acm','ground'):
            server.scene=copy.deepcopy(baseline)
            if kind=='world':
                obj=Object();obj.id='late_part';server.scene.world.collision_objects.append(obj)
            elif kind=='attached':
                obj=Attached();obj.object.id='lego_red_2x3_0';server.scene.robot_state.attached_collision_objects.append(obj)
            elif kind=='acm':server.scene.allowed_collision_matrix.default_entry_names=['*'];server.scene.allowed_collision_matrix.default_entry_values=[True]
            else:
                ground=next(o for o in server.scene.world.collision_objects if o.id=='ground_plane')
                ground.primitive_poses[0].position.z=99.
            with self.subTest(kind=kind):self.assertFalse(c.scene_reset.verify(c.episode_start_physical,JOINTS)['passed'])

    def test_conclusive_false_response_does_not_hide_remote_attachment_from_reset(self):
        c,_,_,server,_=self.setup_scene();server.false_kind='attach'
        c.node._attach_lego_to_gripper('red_2x3_0')
        self.assertIsNone(c.node._attached_lego_name)
        self.assertTrue(server.scene.robot_state.attached_collision_objects)
        self.assertFalse(c.stop.is_set())
        server.false_kind=None
        result=c.scene_reset.clear()
        self.assertEqual(result['removed_attached'],['lego_red_2x3_0'])
        self.assertFalse(server.scene.robot_state.attached_collision_objects)

    def test_reset_refuses_a_mutation_whose_caller_has_not_waited(self):
        c,_,_,server,_=self.setup_scene();server.pending_kind='world_add'
        scene=Scene();obj=Object();obj.id='late_part';scene.world.collision_objects=[obj]
        c.node._apply_scene_client.call_async(Request(scene=scene))
        with self.assertRaises(SceneMutationIncomplete):c.scene_reset.clear()
        self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)

    def test_kit_pose_timeout_is_terminal_and_requires_both_instance_processes(self):
        c,_,_,_,markers=self.setup_scene()
        def timeout(*args,**kwargs):raise TimeoutError('Kit pose completion unknown')
        c.mutate_kit.__func__.__globals__['mcp']=timeout
        with self.assertRaises(TimeoutError):c.mutate_kit('soarm101.randomize_cups',{})
        self.assertEqual(markers[0][1]['source'],'Kit')
        self.assertTrue(c.stop.is_set());self.assertIsNone(c.generation.epoch)
        with self.assertRaises(SceneMutationIncomplete):c.scene_reset.clear()

    def test_startup_pose_callbacks_cannot_overwrite_physical_cache_after_handover(self):
        for callback,lockname,cachekey,key in [('_objects_callback','objects_lock','objects_data','red_2x3_0'),
                                             ('_drop_callback','_drop_lock','_drop_data','drop_3')]:
            c,_,_,_,_=self.setup_scene();n=c.node;entered=threading.Event();release=threading.Event()
            class GateLock:
                def __init__(self):self.inner=threading.RLock()
                def __enter__(self):
                    if threading.current_thread().name=='old-pose':entered.set();release.wait(2)
                    self.inner.acquire();return self
                def __exit__(self,*args):self.inner.release()
            setattr(n,lockname,GateLock());n._physical_pose_cache=False
            function=method(SHARED,'PickPlaceNode',callback)
            tf=NS(child_frame_id=key,transform=NS(translation=NS(x=99.,y=99.,z=99.),rotation=NS(x=0.,y=0.,z=0.,w=1.)))
            thread=threading.Thread(name='old-pose',target=lambda:function(n,NS(transforms=[tf])))
            thread.start();self.assertTrue(entered.wait(2))
            c.update_pose_cache.__func__.__globals__['physical_pose_caches']=__import__('scene_sync').physical_pose_caches
            c.update_pose_cache(self.fixture());physical=copy.deepcopy(getattr(n,cachekey))
            release.set();thread.join(2)
            self.assertFalse(thread.is_alive());self.assertEqual(getattr(n,cachekey),physical)

    def test_old_controller_reference_is_logged_with_source_time_and_unknown_epoch(self):
        c,r,entry,_,_=self.setup_scene();c.timeline(False)
        entry['tick'][0]['base_world_pose']=dict(position=[0.]*3,orientation=[1.,0.,0.,0.])
        Sensors,_=load_class('collect.py','Sensors');sensors=c.sensors
        sensors.generation=c.generation
        sensors.reference_at=MethodType(Sensors.reference_at,sensors)
        msg=NS(header=NS(stamp=NS(sec=1,nanosec=0)),name=JOINTS,position=[.7]*6)
        Sensors.reference_received(sensors,msg)
        rows=[];c.writer=NS(add=lambda row,images:rows.append(row))
        c.capture.__func__.__globals__['Sensors']=NS(image=lambda msg:np.zeros((2,2,3),dtype=np.uint8))
        workers.WorkerRetirementTests().capture(c,r,entry,2,1033333333,False)
        reference=rows[0]['controller_reference']
        self.assertEqual(reference['stamp_ns'],10**9);self.assertEqual(reference['receipt_epoch'],2)
        self.assertIsNone(reference['source_epoch']);self.assertFalse(reference['same_tick'])
        self.assertEqual(reference['validity'],'before_reset_or_unknown')
        self.assertEqual(rows[0]['expert_reference'],[0.]*6)
    def test_reference_callback_does_not_wait_on_generation_ownership(self):
        c,_,_,_,_=self.setup_scene();Sensors,_=load_class('collect.py','Sensors')
        sensors=c.sensors;sensors.generation=c.generation
        msg=NS(header=NS(stamp=NS(sec=1,nanosec=0)),name=JOINTS,position=[0.]*6)
        with c.generation.lock:
            worker=threading.Thread(target=lambda:Sensors.reference_received(sensors,msg))
            worker.start();worker.join(.5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(sensors.reference['source_epoch'],1)



class MemoryPath:
    def __init__(self,key,files):self.key,self.files=key,files
    def exists(self):return self.key in self.files
    def read_text(self):return self.files[self.key]
    def write_text(self,text):self.files[self.key]=text
    def with_suffix(self,suffix):return MemoryPath(self.key+suffix,self.files)
    def replace(self,target):self.files[target.key]=self.files.pop(self.key)


class SceneMarkerTests(unittest.TestCase):
    def test_same_or_unknown_stack_cannot_clear_a_scene_marker(self):
        self.assertFalse(scene_stack_ready(None,{'pid':2}))
        self.assertFalse(scene_stack_ready({'pid':1},None))
        self.assertFalse(scene_stack_ready({'pid':1},{'pid':1}))
        marker=object.__new__(SceneSafetyMarker);marker.runtime=object();marker.path=MemoryPath('marker',{})
        identities={'control.pid':{'pid':1},'isaacsim-soarm101.pid':{'pid':2}}
        with patch('storage.control_identity',side_effect=lambda runtime,pidfile='control.pid':identities[pidfile]):
            marker.record('pending attach')
            with self.assertRaises(RuntimeError):marker.require_fresh_stack()
            identities['control.pid']={'pid':3};marker.require_fresh_stack()
            marker.record('pending pose reset',source='Kit')
            identities['control.pid']={'pid':4}
            with self.assertRaises(RuntimeError):marker.require_fresh_stack()
            identities['isaacsim-soarm101.pid']={'pid':5};marker.require_fresh_stack()
        self.assertEqual(list(marker.path.files),['marker'])


if __name__=='__main__':unittest.main()
