#!/usr/bin/env python3
"""Actual GUI ROS boot against async fake MoveIt; optional real Tk event loop."""
import argparse,copy,json,os,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'vla_SO-ARM101/src/so_arm101_control'))
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.msg import PlanningScene,AllowedCollisionEntry,AttachedCollisionObject,CollisionObject
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from storage import confined,prepare_environment,atomic_json


def operation(value):
    return int.from_bytes(value, "little") if isinstance(value, bytes) else int(value)


class FakeMoveIt(Node):
    def __init__(self):
        super().__init__('fake_moveit_async_scene');self.scene=PlanningScene()
        self.scene.allowed_collision_matrix.entry_names=['base']
        self.scene.allowed_collision_matrix.entry_values=[AllowedCollisionEntry(enabled=[False])]
        self.lock=threading.RLock();self.active=0;self.max_active=0;self.calls=[]
        self.first_gate=threading.Event();group=ReentrantCallbackGroup()
        self.apply=self.create_service(ApplyPlanningScene,'/apply_planning_scene',self.apply_scene,callback_group=group)
        self.get=self.create_service(GetPlanningScene,'/get_planning_scene',self.get_scene,callback_group=group)
    def apply_scene(self,request,response):
        with self.lock:self.active+=1;self.max_active=max(self.active,self.max_active);startup=not self.calls
        if startup:self.first_gate.wait(timeout=5)
        time.sleep(.15) # Wall-time service work runs independently of GUI/ROS caller threads.
        with self.lock:
            scene=request.scene
            self.calls.append(dict(world=[(o.id,operation(o.operation)) for o in scene.world.collision_objects],
                attached=[(o.object.id,operation(o.object.operation)) for o in scene.robot_state.attached_collision_objects],
                acm=bool(scene.allowed_collision_matrix.entry_names)))
            world={o.id:o for o in self.scene.world.collision_objects}
            attached={o.object.id:o for o in self.scene.robot_state.attached_collision_objects}
            for o in scene.robot_state.attached_collision_objects:
                if operation(o.object.operation)==operation(CollisionObject.REMOVE):attached.pop(o.object.id,None)
                else:attached[o.object.id]=copy.deepcopy(o)
            for o in scene.world.collision_objects:
                if operation(o.operation)==operation(CollisionObject.REMOVE):world.pop(o.id,None)
                else:world[o.id]=copy.deepcopy(o)
            self.scene.world.collision_objects=list(world.values())
            self.scene.robot_state.attached_collision_objects=list(attached.values())
            if scene.allowed_collision_matrix.entry_names:self.scene.allowed_collision_matrix=copy.deepcopy(scene.allowed_collision_matrix)
            self.active-=1
        response.success=True;return response
    def get_scene(self,request,response):
        with self.lock:response.scene=copy.deepcopy(self.scene)
        return response


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--tk',action='store_true');args=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise ValueError('Use assigned isolated domain90, no live sim domain')
    prepare_environment(0);output=confined(args.output);rclpy.init()
    from so_arm101_control.control_gui import SOArm101ControlGUI
    from so_arm101_control.scene_transactions import SceneMutationIncomplete,SceneMutationClient
    server=FakeMoveIt();gui=SOArm101ControlGUI();client=Node('gui_scene_regression_client')
    for name in ('execution','_motion_effect_guard','_scene_failure_handler','_callback_failure_handler'):
        try:setattr(gui,name,None)
        except AttributeError:pass
        else:raise AssertionError('Construction policy is mutable: '+name)
    assert gui.execution.mode=='gui' and not isinstance(gui._apply_scene_client,SceneMutationClient)
    assert not hasattr(gui,'_repair_gui_scene') and not hasattr(gui._apply_scene_client,'gui')
    workers=[];worker_errors=[];original_thread=gui._motion_thread
    def tracked(target,**kwargs):
        def checked():
            try:target()
            except Exception as exc:worker_errors.append(repr(exc))
        thread=original_thread(target=checked,**kwargs);workers.append(thread);return thread
    gui._motion_thread=tracked
    executor=MultiThreadedExecutor(num_threads=6)
    for node in [server,gui,client]:executor.add_node(node)
    executor_errors=[];heartbeat=[0]
    def spin():
        try:executor.spin()
        except BaseException as exc:executor_errors.append(repr(exc))
    ros_thread=threading.Thread(target=spin);ros_thread.start()
    timer=gui.create_timer(.02,lambda:heartbeat.__setitem__(0,heartbeat[0]+1))
    root=None;receipt=dict(passed=False,mode='GUI raw services/background dispatch + async fake MoveIt',tk=args.tk)
    def wait_for(condition,seconds=10):
        end=time.monotonic()+seconds
        while not condition():
            if time.monotonic()>end:raise TimeoutError('GUI regression condition did not complete')
            threading.Event().wait(.01)
    def probes():
        try:
            gui._drop_data={f'drop_{i}':dict(x=.10*i,y=-.2,z=.04825,qw=1.) for i in range(1,4)}
            wait_for(lambda:gui._apply_scene_client.wait_for_service(timeout_sec=.01))
            started=time.monotonic();gui._cmd_toggle_ground_plane();gui._add_cup_collision_objects()
            dispatch_seconds=time.monotonic()-started
            wait_for(lambda:server.active>=2);server.first_gate.set()
            wait_for(lambda:all(not w.is_alive() for w in workers) and server.active==0)
            assert not worker_errors and not getattr(gui,'_scene_terminal_error',None)
            assert server.max_active>=2 # No collector queue/serial ownership on the GUI path.
            before=heartbeat[0]
            def bad_goal():raise SceneMutationIncomplete('joint callback guard regression')
            gui._publish_goal_state=bad_goal
            pub=client.create_publisher(JointState,'/joint_states',10);wait_for(lambda:pub.get_subscription_count()>0)
            pub.publish(JointState(name=list(gui.joint_positions),position=[0.]*len(gui.joint_positions)))
            def bad_service(request,response):raise SceneMutationIncomplete('service guard regression')
            probe=gui.create_service(Trigger,'~/guard_probe',bad_service)
            call=client.create_client(Trigger,'/so_arm101_control_gui/guard_probe');wait_for(lambda:call.wait_for_service(timeout_sec=.01))
            future=call.call_async(Trigger.Request());wait_for(future.done);assert future.result().success is False
            def bad_timer():raise SceneMutationIncomplete('timer guard regression')
            bad=gui.create_timer(.02,bad_timer)
            wait_for(lambda:len(getattr(gui,'_ros_callback_errors',[]))>=3);bad.cancel()
            good=client.create_client(Trigger,'/so_arm101_control_gui/get_joint_positions');wait_for(lambda:good.wait_for_service(timeout_sec=.01))
            future=good.call_async(Trigger.Request());wait_for(future.done);assert future.result().success
            wait_for(lambda:heartbeat[0]>before+2)
            if args.tk:
                # Reproduce the reviewer's Refresh entry on the actual Tk thread.
                # Native detach waits on ROS inline; its real _append_log.after
                # must not be routed through a scene worker that waits on Tk.
                gui._publish_goal_state=lambda:None
                gui._attached_lego_name='red_2x3_0';gui._attached_lego_tcp_offset=(0.,0.,0.)
                obj=AttachedCollisionObject();obj.link_name='tcp_link';obj.object.id='lego_red_2x3_0'
                with server.lock:server.scene.robot_state.attached_collision_objects=[obj]
                done=threading.Event();tk_errors=[];ticks=receipt['tk_heartbeat']
                before_ticks=ticks[0]
                def refresh():
                    started=time.monotonic()
                    try:gui._cmd_grasp_refresh()
                    except Exception as exc:tk_errors.append(repr(exc))
                    finally:
                        receipt['tk_refresh_wall_seconds']=time.monotonic()-started
                        done.set()
                gui.root.after(0,refresh);wait_for(done.is_set,seconds=5)
                wait_for(lambda:ticks[0]>before_ticks+2)
                assert not tk_errors and gui._attached_lego_name is None
                with server.lock:assert not server.scene.robot_state.attached_collision_objects
                receipt.update(tk_refresh_completed=True,tk_refresh_errors=tk_errors,tk_heartbeat_after=ticks[0])
            wait_for(lambda:all(not w.is_alive() for w in workers) and server.active==0)
            assert not worker_errors
            receipt.update(passed=True,explicit_mode=gui.execution.mode,collector_wrapper=False,scene_worker=False,
                resync=False,raw_apply_max_concurrency=server.max_active,startup_dispatch_seconds=dispatch_seconds,
                completed_apply_calls=server.calls,callback_errors=gui._ros_callback_errors,executor_alive=ros_thread.is_alive(),
                heartbeat=heartbeat[0],query_service_alive=True,scene_terminal_error=getattr(gui,'_scene_terminal_error',None),
                worker_errors=worker_errors)
        except BaseException as exc:receipt['error']=repr(exc)
        finally:
            server.first_gate.set()
            if root is not None:root.after(0,root.quit)
    try:
        if args.tk:
            import tkinter as tk
            root=tk.Tk();root.withdraw();gui.root=root;gui._gui_ready=True
            gui.obj_listbox=tk.Listbox(root);gui._sim_ensure_sim_topics=lambda:None
            ticks=[0];receipt['tk_heartbeat']=ticks
            def beat():ticks[0]+=1;root.after(20,beat)
            root.after(20,beat);probe_thread=threading.Thread(target=probes);root.after(0,probe_thread.start)
            root.mainloop();probe_thread.join(timeout=5);assert not probe_thread.is_alive()
        else:
            gui._gui_ready=True;gui.root=NS(after=lambda *a:None)
            # Exercise the real GUI log body; only the display adapter is absent.
            probes()
    finally:
        server.first_gate.set();timer.cancel();gui._gui_ready=False;gui.running=False
        for worker in workers:worker.join(timeout=5)
        executor.shutdown(timeout_sec=5);ros_thread.join(timeout=5)
        for node in [gui,server,client]:node.destroy_node()
        rclpy.shutdown()
        if root is not None:root.destroy()
        receipt.update(executor_thread_stopped=not ros_thread.is_alive(),executor_errors=executor_errors,
                       scene_threads_stopped=all(not w.is_alive() for w in workers))
        atomic_json(output,receipt);print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise RuntimeError(receipt.get('error','GUI regression failed'))
if __name__=='__main__':main()
