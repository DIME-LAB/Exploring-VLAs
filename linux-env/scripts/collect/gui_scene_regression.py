#!/usr/bin/env python3
"""Offline ROS GUI regression: fake MoveIt, delayed responses, live executor; no Isaac or display."""
import argparse,copy,json,os,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'vla_SO-ARM101/src/so_arm101_control'))
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.msg import PlanningScene,AllowedCollisionEntry
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from storage import confined,prepare_environment

class FakeMoveIt(Node):
    def __init__(self):
        super().__init__('fake_moveit_async_scene');self.scene=PlanningScene();self.scene.allowed_collision_matrix.entry_names=['base']
        self.scene.allowed_collision_matrix.entry_values=[AllowedCollisionEntry(enabled=[False])]
        self.lock=threading.RLock();self.active=0;self.max_active=0;self.calls=[]
        self.first_gate=threading.Event();group=ReentrantCallbackGroup()
        self.apply=self.create_service(ApplyPlanningScene,'/apply_planning_scene',self.apply_scene,callback_group=group)
        self.get=self.create_service(GetPlanningScene,'/get_planning_scene',self.get_scene,callback_group=group)
    def apply_scene(self,request,response):
        with self.lock:
            self.active+=1;self.max_active=max(self.active,self.max_active);first=not self.calls
        if first:self.first_gate.wait(timeout=5)
        time.sleep(5.2 if first else .05) # First ground ADD exceeds its actual 5 s client wait.
        with self.lock:
            scene=request.scene
            self.calls.append(dict(world=[o.id for o in scene.world.collision_objects],acm=bool(scene.allowed_collision_matrix.entry_names)))
            world={o.id:o for o in self.scene.world.collision_objects}
            for o in scene.world.collision_objects:
                if o.operation==o.REMOVE:world.pop(o.id,None)
                else:world[o.id]=copy.deepcopy(o)
            self.scene.world.collision_objects=list(world.values())
            if scene.allowed_collision_matrix.entry_names:self.scene.allowed_collision_matrix=copy.deepcopy(scene.allowed_collision_matrix)
            self.active-=1
        response.success=True;return response
    def get_scene(self,request,response):
        with self.lock:response.scene=copy.deepcopy(self.scene)
        return response

def wait_for(condition,seconds=10):
    end=time.monotonic()+seconds
    while not condition():
        if time.monotonic()>end:raise TimeoutError('Offline GUI regression condition did not complete')
        threading.Event().wait(.01)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise ValueError('Use assigned isolated domain90, no live sim domain')
    prepare_environment(0);output=confined(args.output);rclpy.init()
    from so_arm101_control.control_gui import SOArm101ControlGUI
    from so_arm101_control.scene_transactions import SceneMutationIncomplete
    server=FakeMoveIt();gui=SOArm101ControlGUI();client=Node('gui_scene_regression_client')
    for name in ('execution','_motion_effect_guard','_scene_failure_handler','_callback_failure_handler'):
        try:setattr(gui,name,None)
        except AttributeError:pass
        else:raise AssertionError('Construction policy is mutable: '+name)
    # Boot the actual GUI node/registrations with a display-free UI adapter.
    gui._gui_ready=True;gui.root=NS(after=lambda *a:None)
    gui._append_log=lambda text,level='info':gui.get_logger().info(text)
    gui._drop_data={f'drop_{i}':dict(x=.10*i,y=-.2,z=.04825,qw=1.) for i in range(1,4)}
    executor=MultiThreadedExecutor(num_threads=6)
    for node in [server,gui,client]:executor.add_node(node)
    failure=[]
    def spin():
        try:executor.spin()
        except BaseException as exc:failure.append(repr(exc))
    thread=threading.Thread(target=spin);thread.start();heartbeat=[0]
    timer=gui.create_timer(.02,lambda:heartbeat.__setitem__(0,heartbeat[0]+1))
    receipt=dict(passed=False,mode='Actual GUI ROS node + fake MoveIt delayed services; fake UI adapter, no Isaac/display')
    try:
        wait_for(lambda:gui._apply_scene_client.wait_for_service(timeout_sec=.01))
        # These used to overlap at startup and terminally poison the GUI.
        gui._cmd_toggle_ground_plane();gui._add_cup_collision_objects()
        wait_for(lambda:server.active==1)
        assert len(gui._apply_scene_client.gui.jobs)>=1
        helper_done=threading.Event()
        helper=threading.Thread(target=lambda:(gui._acm_allow_ground_rest(['lego_red_2x3_0']),helper_done.set()))
        helper.start()
        server.first_gate.set()
        wait_for(lambda:len(server.calls)>=5 and not gui._apply_scene_client.gui.has_work())
        helper.join(timeout=2);assert helper_done.is_set()
        assert not gui._apply_scene_client.gui.needs_resync and gui._gui_scene_repair_receipt
        assert server.max_active==1 and not gui._apply_scene_client.failure
        assert not getattr(gui,'_scene_terminal_error',None)
        assert 'ground_plane' in server.scene.allowed_collision_matrix.entry_names
        assert 'lego_red_2x3_0' in server.scene.allowed_collision_matrix.entry_names
        assert sum(n.startswith('cup_') for n in server.scene.allowed_collision_matrix.entry_names)==3
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
        assert thread.is_alive() and not failure and not getattr(gui,'_scene_terminal_error',None)
        receipt.update(passed=True,first_apply_delay_seconds=5.2,authoritative_repair=gui._gui_scene_repair_receipt,
          direct_attach_style_acm_permission_preserved=True,explicit_mode=gui.execution.mode,overlapping_startup_requests=2,caller_overlap_observed=True,raw_apply_max_concurrency=server.max_active,completed_apply_calls=server.calls,
          callback_errors=gui._ros_callback_errors,executor_alive=thread.is_alive(),heartbeat=heartbeat[0],query_service_alive=True,
          scene_terminal_error=getattr(gui,'_scene_terminal_error',None),strict_client_failure=gui._apply_scene_client.failure)
    finally:
        server.first_gate.set();timer.cancel();gui._gui_ready=False;gui.running=False
        executor.shutdown(timeout_sec=5);thread.join(timeout=5)
        for node in [gui,server,client]:node.destroy_node()
        rclpy.shutdown();receipt['executor_thread_stopped']=not thread.is_alive();receipt['executor_errors']=failure
        output.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
