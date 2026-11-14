#!/usr/bin/env python3
"""Construct the actual collector and reach a contract-valid held tick without Isaac."""
import argparse,json,os,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from storage import confined,prepare_environment,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise ValueError('Use isolated domain 90')
    runtime=prepare_environment(0)
    from run_smoke import InstanceClaim
    instance_claim=InstanceClaim(runtime)
    import collect as implementation
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState,Image
    from std_msgs.msg import String
    from moveit_msgs.srv import GetStateValidity
    from contract import JOINTS
    from protocol import CancelledGeneration
    from tick_contract import require_snapshots
    from cup_geometry import PROFILE
    output=confined(args.output);output.mkdir(parents=True,exist_ok=False)
    rclpy.init()
    from rclpy.context import Context
    native_context=Context();native_context.init()
    native=Node('fake_collector_boot_transport',context=native_context)
    clock=native.create_publisher(Clock,'/clock',10)
    joint=native.create_publisher(JointState,'/joint_states_measured',10)
    cameras=[native.create_publisher(Image,name,10) for name in ('/wrist_camera_rgb_sim','/workspace_camera_sim')]
    tick=native.create_publisher(String,'/collection_tick',10)
    ack=native.create_publisher(String,'/collection_command_ack',10)
    commands=[]
    def applied(msg):
        payload=json.loads(msg.data);commands.append(payload)
        ack.publish(String(data=json.dumps(dict(epoch=payload['epoch'],observation_stamp_ns=payload['observation_stamp_ns'],
            application_stamp_ns=payload['observation_stamp_ns'],applied_command=payload['command']))))
    subscription=native.create_subscription(String,'/collection_command',applied,10)
    queries=[]
    def valid(request,response):queries.append(list(request.robot_state.joint_state.position));response.valid=True;return response
    service=native.create_service(GetStateValidity,'/check_state_validity',valid)
    executor=MultiThreadedExecutor(num_threads=3,context=native_context);executor.add_node(native)
    thread=threading.Thread(target=executor.spin);thread.start()
    collector=None;publishers=[];result=dict(passed=False,actual_collector_constructor=True,actual_pick_place_constructor=True)
    stamp=1033333333;frame=dict(epoch=1,sim_ns=stamp,physics_time_s=stamp/1e9,state=[0.]*6,velocity=[0.]*6,
        drive_targets=[0.]*6,bodies={},base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),state_source='isaac_articulation')
    class HeldTickReached(Exception):pass
    def fake_mcp(tool,params=None):
        code=(params or {}).get('code','')
        if 'result=configure(30)' in code:return dict(fps=30)
        if 'result=c.install()' in code:
            offsets={name:{key+'_m':[PROFILE['expected_offsets_m']['cup_'+key]]*len(PROFILE['primitives']) for key in ('contact','rest')}
                     for name in ('cup_red','cup_green','cup_blue')}
            return dict(epoch=1,invalid_through_ns=1000000000,local_corners={},effective_shape_offsets=offsets)
        if 'count_subscribers' in code:return [2,1,1]
        if 'rateLimitFrequency' in code:return True
        raise AssertionError('Unexpected fake Kit request: '+code)
    def native_tick(play):
        if not play:return dict(held=True,failure=None,epoch=2,invalid_through_ns=stamp,
                                clock_baseline=dict(epoch=2,sim_ns=stamp,physics_time_s=stamp/1e9))
        def publish_frame():
            end=time.monotonic()+5
            while min([joint.get_subscription_count(),tick.get_subscription_count(),ack.get_subscription_count(),
                       *[p.get_subscription_count() for p in cameras]])<1:
                if time.monotonic()>end:raise TimeoutError('Fake native publisher discovery failed')
                threading.Event().wait(.01)
            c=Clock();c.clock.sec=1;c.clock.nanosec=33333333;clock.publish(c)
            j=JointState();j.header.stamp.sec=1;j.header.stamp.nanosec=33333333;j.name=list(JOINTS);j.position=[0.]*6;j.velocity=[0.]*6;joint.publish(j)
            for pub in cameras:
                msg=Image();msg.header.stamp.sec=1;msg.header.stamp.nanosec=33333500;msg.encoding='rgb8';msg.width=640;msg.height=480;msg.step=1920;msg.data=bytes(921600);pub.publish(msg)
            tick.publish(String(data=json.dumps(frame)))
        t=threading.Thread(target=publish_frame);publishers.append(t);t.start()
        return dict(start_ns=966666667,target_ns=stamp)
    def first_held_tick(self):
        assert self.last_snapshot['sim_ns']==stamp
        assert require_snapshots([self.last_snapshot])['passed']
        assert commands and not commands[-1]['advance']
        assert queries and self.generation.epoch==1 and self.failure is None
        assert self._constructing_thread is None
        with self.motion_effect(1):pass
        try:
            with self.motion_effect(None):raise AssertionError('Unowned effect accepted after bootstrap')
        except CancelledGeneration:pass
        result.update(passed=True,first_tick=self.last_snapshot,commands=commands,validity_queries=len(queries),
                      bootstrap_revoked=True,camera_native_skew_ns=167,wall_watchdog_margin_s=self.node._motion_waiter.wall_margin,executor_alive=self.ros_thread.is_alive(),capture_alive=self.capture_thread.is_alive())
        raise HeldTickReached()
    try:
        config=NS(output=output/'run',episodes=5,max_attempts=1,resume=False,seed=1,regrasp_only=False,normal_only=True,replay_receipt=None,wait_rtf_floor=.1,episode_deadline=180,diagnostic=True,instance=0)
        with patch.object(implementation,'mcp',fake_mcp),patch.object(implementation,'timeline',native_tick),\
             patch.object(implementation.Collector,'wait_ready',first_held_tick):
            try:
                collector=implementation.Collector(config)
                # Before run installs an epoch the constructor-only allowance has already expired.
                try:
                    with collector.motion_effect(None):raise AssertionError('Bootstrap leaked before run')
                except CancelledGeneration:pass
                collector.run()
                raise AssertionError('run did not reach the first held tick')
            except HeldTickReached:pass
            finally:
                if collector is not None:
                    # Keep fake transport's executor alive until the collector has retired.
                    collector.close()
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        for worker in publishers:worker.join(timeout=6)
        from ros_lifecycle import retire_executor,shutdown_context
        retire_executor(executor,thread);native.destroy_node();shutdown_context(rclpy,native_context)
        if rclpy.ok():rclpy.shutdown()
        instance_claim.close()
        atomic_json(output/'validation.json',result)
    print(json.dumps(dict(passed=result['passed'],validity_queries=result['validity_queries'],
        tick_ns=result['first_tick']['sim_ns'],output=str(output)),indent=2))


if __name__=='__main__':main()
