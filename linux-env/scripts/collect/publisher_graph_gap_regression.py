from pathlib import Path
import sys,os,time,json,argparse
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'linux-env/scripts/collect'))
from storage import prepare_environment,atomic_json
prepare_environment(0)
if os.environ.get('ROS_DOMAIN_ID')!='90':raise RuntimeError('Use the assigned isolated domain 90')
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
parser=argparse.ArgumentParser(description='Show a transient same-format publisher missed by 1 s graph polling; no Isaac')
parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
from storage import confined
out=confined(args.output);out.mkdir(exist_ok=False)
rclpy.init();receiver=rclpy.create_node('graph_gap_receiver');a=rclpy.create_node('graph_gap_a');b=rclpy.create_node('graph_gap_b')
topic='/collection_graph_gap_probe_rgb';pub_a=a.create_publisher(Image,topic,10);rx=[];checks=[];start=time.monotonic();pub_b=None
executor=SingleThreadedExecutor();executor.add_node(receiver)
def ids():return sorted(bytes(row.endpoint_gid).hex() for row in receiver.get_publishers_info_by_topic(topic))
def receive(msg):rx.append(dict(value=int(msg.data[0]),stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,encoding=msg.encoding,shape=[msg.width,msg.height],wall_s=time.monotonic()-start))
sub=receiver.create_subscription(Image,topic,receive,10)
def until(predicate,seconds):
 deadline=time.monotonic()+seconds
 while not predicate():
  if time.monotonic()>deadline:raise TimeoutError('Graph-gap probe discovery/delivery deadline')
  executor.spin_once(timeout_sec=.005)
def frame(value):
 msg=Image();msg.header.stamp.sec=2;msg.encoding='rgb8';msg.width=640;msg.height=480;msg.step=1920;msg.data=bytes([value])*921600;return msg
result=dict(conclusive=False,interval_s=1.,no_isaac=True)
try:
 until(lambda:pub_a.get_subscription_count()==1 and len(ids())==1,2.)
 baseline=ids();start=time.monotonic();checks.append(dict(wall_s=0.,gids=baseline))
 timer=receiver.create_timer(1.,lambda:checks.append(dict(wall_s=time.monotonic()-start,gids=ids())))
 pub_b=b.create_publisher(Image,topic,10)
 until(lambda:pub_b.get_subscription_count()==1 and len(ids())==2,.65)
 during=ids();pub_b.publish(frame(222));until(lambda:any(row['value']==222 for row in rx),.2)
 b.destroy_publisher(pub_b);pub_b=None
 pub_a.publish(frame(11));until(lambda:any(row['value']==11 for row in rx),.2)
 duration=time.monotonic()-start
 until(lambda:len(checks)>=3,2.5)
 result.update(conclusive=duration<1. and all(row['gids']==baseline for row in checks) and rx[0]['value']==222,
   transient_window_s=duration,baseline_gids=baseline,during_gids=during,periodic_checks=checks,messages=rx,
   first_frame_value=rx[0]['value'],first_frame_actual_publisher='transient B',
   explanation='Second rgb8 publisher sent the first image at the identical sim stamp and vanished between all 1 s graph checks')
finally:
 executor.shutdown(timeout_sec=2);receiver.destroy_node();a.destroy_node();b.destroy_node();rclpy.shutdown()
 atomic_json(out/'validation.json',result)
print(json.dumps(result,indent=2))
if not result['conclusive']:raise RuntimeError('Probe did not establish a graph-check gap; do not infer a result')
