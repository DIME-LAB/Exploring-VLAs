#!/usr/bin/env python3
"""Real Humble image packets: GID preservation, repeat renders and dual sources; no Isaac."""
import argparse,hashlib,json,os,sys,time,threading
from pathlib import Path
from storage import prepare_environment,confined,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise RuntimeError('Use assigned isolated domain 90')
    runtime=prepare_environment(0)
    from run_smoke import InstanceClaim
    from managed_process import verified
    instance_claim=InstanceClaim(runtime)
    for name in ('isaacsim-soarm101.pid','control.pid'):
        if verified(runtime/name):raise RuntimeError('Identity probe requires its private instance to be down')
    root=confined(args.output);root.mkdir(parents=True,exist_ok=False)
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image,JointState
    from std_msgs.msg import String
    from so_arm101_control.ros_callbacks import safe_callback
    from image_transport import publisher_executor_type
    from collect import Sensors
    from protocol import StepGeneration
    from frame_join import SensorAmbiguity
    from contract import JOINTS
    errors=[]
    class Probe(Node):
        def create_subscription(self,kind,topic,callback,qos,**kwargs):
            return super().create_subscription(kind,topic,safe_callback(callback,lambda name,error:errors.append(dict(name=name,error=str(error)))),qos,**kwargs)
    rclpy.init();node=Probe('publisher_identity_probe');a=Node('publisher_identity_a');b=Node('publisher_identity_b')
    generation=StepGeneration();floor=178316666667;generation.install(dict(epoch=1,invalid_through_ns=floor))
    sensors=Sensors(node,generation);executor=publisher_executor_type()(num_threads=1);executor.add_node(node)
    pa=a.create_publisher(Image,'/wrist_camera_rgb_sim',64);pt=a.create_publisher(Image,'/workspace_camera_sim',64)
    pj=a.create_publisher(JointState,'/joint_states_measured',64);pk=a.create_publisher(String,'/collection_tick',64)
    result=dict(passed=False,no_isaac=True,native_adapter=executor.image_take.metadata)
    def until(predicate,seconds=5):
        deadline=time.monotonic()+seconds
        while not predicate():
            if time.monotonic()>deadline:raise TimeoutError('Publisher identity probe deadline')
            executor.spin_once(timeout_sec=.01)
    def image(stamp,value):
        msg=Image();msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(stamp,10**9);msg.header.frame_id='camera_link'
        msg.encoding='rgb8';msg.width=640;msg.height=480;msg.step=1920;msg.data=bytes([value])*921600;return msg
    try:
        until(lambda:all(p.get_subscription_count()==1 for p in (pa,pt,pj,pk)))
        pa.publish(image(floor-1,7));pa.publish(image(floor-1,8))
        until(lambda:sensors.stale_camera_messages.get('wrist',0)==2)
        assert not sensors.failure and not sensors.camera_publishers
        stamp=floor+33333333
        first_stamp=stamp-2
        pa.publish(image(first_stamp,11));pt.publish(image(stamp,33))
        j=JointState();j.header.stamp.sec,j.header.stamp.nanosec=divmod(stamp,10**9);j.name=list(JOINTS);j.position=[0.]*6;j.velocity=[0.]*6
        pj.publish(j);pk.publish(String(data=json.dumps(dict(epoch=1,sim_ns=stamp))))
        until(lambda:any(len(entry)==4 for entry in sensors.pending.values()))
        selected,entry=sensors.read(timeout=0);assert selected==first_stamp and entry['wrist'][0].data[0]==11
        gid=entry['wrist'][2]['publisher']['gid'];known=[bytes(row.endpoint_gid).hex() for row in node.get_publishers_info_by_topic('/wrist_camera_rgb_sim')]
        result['packet_identity']=entry['wrist'][2]['publisher'];result['graph_endpoint_gids']=known
        pa.publish(image(stamp-1,22));pt.publish(image(stamp+1,44));until(lambda:sum(sensors.rerender_messages.values())==2)
        assert not sensors.failure and entry['wrist'][0].data[0]==11 and entry['top'][0].data[0]==33
        decode=[]
        for index in range(1,51):
            next_stamp=stamp+index*33333333;pa.publish(image(next_stamp,11))
            until(lambda:next_stamp//1000 in sensors.seen_camera_frames['wrist'])
            decode.append(sensors.seen_camera_frames['wrist'][next_stamp//1000]['publisher']['transport_take_decode_ms'])
        result['warm_native_take_decode_ms']=dict(frames=len(decode),mean=sum(decode)/len(decode),p95=sorted(decode)[47],maximum=max(decode))
        pb=b.create_publisher(Image,'/wrist_camera_rgb_sim',64);until(lambda:pb.get_subscription_count()==1)
        pb.publish(image(first_stamp,11));until(lambda:sensors.failure is not None)
        assert isinstance(sensors.failure,SensorAmbiguity)
        candidates=sensors.failure.evidence['candidates'];assert len({row['publisher']['gid'] for row in candidates})==2
        assert candidates[0]['content_sha256']==candidates[1]['content_sha256']
        result.update(passed=True,selected=dict(stamp_ns=first_stamp,native_tick=entry['wrist'][2]['native_tick'],wrist_value=int(entry['wrist'][0].data[0]),top_value=int(entry['top'][0].data[0]),
            identities={k:entry[k][2] for k in ('wrist','top')}),
            stale_boundary_frames=dict(sensors.stale_camera_messages),rerenders=dict(sensors.rerender_messages),
            rerender_log=list(sensors.rerender_log.values()),duplicate_publisher=sensors.failure.evidence,
            contained_callback_errors=errors,packet_gid_stable_and_distinct=True,graph_gid_matches_actual_packet=known==[gid])
    finally:
        result['executor_retired']=executor.shutdown(timeout_sec=5)
        node.destroy_node();a.destroy_node();b.destroy_node();rclpy.shutdown()
        atomic_json(root/'validation.json',result);instance_claim.close()
    print(json.dumps(dict(passed=result['passed'],stale=result['stale_boundary_frames'],rerenders=result['rerenders'],output=str(root)),indent=2))


if __name__=='__main__':main()
