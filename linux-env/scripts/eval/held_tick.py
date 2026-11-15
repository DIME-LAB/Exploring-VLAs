"""Collector command/ACK adapter for evaluation while physics waits for each action.

No ROS trajectory controller is used. The installed collector owns native target
application and one-frame advancement; this client sends its existing protocol.
"""
from __future__ import annotations
import math
import struct
import time

STEP_NS=1_000_000_000/30

def float32(values):
    values=list(values)
    if len(values)!=6 or not all(math.isfinite(v) for v in values):
        raise ValueError('A held tick requires six finite absolute joint targets in radians')
    try:result=[struct.unpack('f',struct.pack('f',v))[0] for v in values]
    except (OverflowError,struct.error) as error:raise ValueError('Joint target is outside float32 range') from error
    if not all(math.isfinite(v) for v in result):raise ValueError('Joint target is outside float32 range')
    return result

def verify_ack(payload,ack):
    if 'error' in ack:raise RuntimeError('Collector refused command: '+ack['error'])
    if ack['epoch']!=payload['epoch'] or ack['observation_stamp_ns']!=payload['observation_stamp_ns']:
        raise RuntimeError('Collector ACK belongs to a different observation')
    if abs(ack['application_stamp_ns']-payload['observation_stamp_ns'])>1000:
        raise RuntimeError('Native command was not applied at the held observation tick')
    expected=float32(payload['command']);actual=ack['applied_command']
    if len(actual)!=6 or not all(math.isfinite(v) for v in actual) or float32(actual)!=expected:
        raise RuntimeError('Native drive read-back differs from the command')

def verify_step(previous,next_snapshot,epoch):
    if next_snapshot['epoch']!=epoch:raise RuntimeError('Native frame belongs to a different reset epoch')
    if abs((next_snapshot['sim_ns']-previous['sim_ns'])-STEP_NS)>1000:
        raise RuntimeError('Collector did not advance exactly one 30 Hz frame')

class HeldTick:
    """Own one current joined observation and execute exactly one command per frame."""
    def __init__(self,generation,sensors,publish):
        self.generation=generation;self.sensors=sensors;self.publish=publish
        self.current=None;self.entry=None;self.failed_epoch=None
    def read(self,timeout=30):
        stamp,entry=self.sensors.read(timeout=timeout);snapshot=entry['tick'][0]
        self.generation.require(snapshot['epoch'],stamp)
        if snapshot['epoch']==self.failed_epoch:raise RuntimeError('Failed command exchange requires a new reset epoch')
        if abs(snapshot['sim_ns']-stamp)>1000:raise RuntimeError('Camera/native clock mismatch')
        self.current=snapshot;self.entry=entry
        return snapshot
    def step(self,command):
        if self.current is None:raise RuntimeError('Read a joined held observation before stepping')
        command=float32(command)
        with self.generation.own('eval held action'):
            previous=self.current;epoch=previous['epoch'];stamp=previous['sim_ns']
            self.generation.require(epoch,stamp)
            payload=dict(epoch=epoch,observation_stamp_ns=stamp,command=command,advance=True)
            started=time.monotonic_ns();self.current=None
            try:
                self.publish(payload)
                ack=self.sensors.read_ack(epoch,stamp);verify_ack(payload,ack)
                self.generation.require(epoch,stamp)
                next_snapshot=self.read();verify_step(previous,next_snapshot,epoch)
            except BaseException:
                self.failed_epoch=epoch;self.current=None
                raise
            return dict(command=payload,ack=ack,wall_s=(time.monotonic_ns()-started)/1e9),next_snapshot

class CollectorTransport:
    """Use the qualified collector's sensor join, packet identity and executor."""
    def __init__(self):
        import rclpy
        from rclpy.parameter import Parameter
        from std_msgs.msg import String
        from collect import Sensors
        from protocol import StepGeneration
        from image_transport import publisher_executor_type
        import threading
        self.rclpy=rclpy;self.String=String
        rclpy.init();self.node=rclpy.create_node('eval_collector_held_tick',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.generation=StepGeneration();self.sensors=Sensors(self.node,self.generation)
        self.publisher=self.node.create_publisher(String,'/collection_command',64)
        self.executor=publisher_executor_type()(num_threads=4);self.executor.add_node(self.node)
        self.thread=threading.Thread(target=self.executor.spin,name='collector_eval_ros',daemon=True);self.thread.start()
        self.session=HeldTick(self.generation,self.sensors,lambda payload:self.publisher.publish(String(data=__import__('json').dumps(payload))))
    def close(self):
        from ros_lifecycle import retire_executor,shutdown_context
        self.sensors.closed=True
        with self.sensors.condition:self.sensors.condition.notify_all()
        retired=retire_executor(self.executor,self.thread)
        context=self.node.context;self.node.destroy_node();shutdown_context(self.rclpy,context)
        return retired

POLICY_FEATURES=('shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos')
def validate_policy_features(features):
    if list(features)!=['observation.state','observation.images.top','observation.images.wrist']:raise ValueError('Policy image feature order differs')
    if features['observation.state']['names']!=list(POLICY_FEATURES) or list(features['observation.state']['shape'])!=[6]:raise ValueError('Policy six-joint state order differs')
    if any(list(features[k]['shape'])!=[480,640,3] for k in ('observation.images.top','observation.images.wrist')):raise ValueError('Policy camera geometry differs')
def validate_policy_requests(requests,rtc):
    if not requests:raise ValueError('No policy inference requests')
    for index,request in enumerate(requests):
        trace=request['native_rtc']
        if request['measured_prediction_sim_s']!=0 or trace['inference_delay']!=0:raise ValueError('Physics advanced during held inference')
        if trace['enabled']!=rtc:raise ValueError('RTC mode differs')
        if rtc and (trace['execution_horizon']!=10 or trace['max_guidance_weight']!=10 or trace['prefix_steps']!=(0 if index==0 else 40)):raise ValueError('RTC guidance contract differs')
