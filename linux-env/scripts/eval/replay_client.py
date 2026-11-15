"""Episode-gated recorded actions through the production SO101ROS2 backend."""
import argparse,json,math,os,sys,time
from pathlib import Path
import draccus
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import make_robot_from_config
from sim_time import PacingClock,TickSchedule
from actuation import install

def emit(tag,data):print(tag+' '+json.dumps(data),flush=True)
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--recorded-replay',type=Path,required=True);p.add_argument('--actuation-mode',choices=('native','immediate'),default='native');a,remaining=p.parse_known_args();cfg=draccus.parse(RobotClientConfig,args=remaining)
 assert os.environ.get('ROS_DOMAIN_ID')=='92';cfg.robot.cameras={}
 robot=make_robot_from_config(cfg.robot);robot.connect();clock=PacingClock(True,robot._node);clock.wait_ready();context={};published=[]
 def witness(label,msg):
  published.append(dict(context,controller=label,joint_names=list(msg.joint_names),positions_rad=list(msg.points[0].positions),time_from_start_s=msg.points[0].time_from_start.sec+msg.points[0].time_from_start.nanosec/1e9,header_stamp_ns=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec))
 restore=install(robot,a.actuation_mode,True,witness);episodes=json.loads(a.recorded_replay.read_text());emit('READY',dict(backend=type(robot).__name__,mode=a.actuation_mode,use_degrees=robot.config.use_degrees,actuate=robot.config.actuate,joint_names=robot.config.joint_names))
 try:
  for line in sys.stdin:
   parts=line.split()
   if not parts:continue
   if parts[0] in ('QUIT','ABORT'):break
   if parts[0]!='RUN':raise ValueError('Unexpected replay command')
   ep=parts[1];record=episodes[ep];actions=record['actions_degrees'];published.clear();rows=[];start=clock.now();schedule=TickSchedule(start,30);missed=0
   emit('EPSTART',dict(ep=ep,clock_s=start,actions=len(actions)))
   import threading
   stop=threading.Event()
   for i,values in enumerate(actions):
    now=clock.wait_until(schedule.deadline,stop);expected,skip=schedule.take(now);missed+=skip
    context.clear();context.update(action_index=i,clock_s=now,deadline_s=expected)
    action={name+'.pos':float(value) for name,value in zip(robot.config.joint_names,values)}
    before=time.perf_counter();sent=robot.send_action(action);cost=(time.perf_counter()-before)*1000
    rows.append(dict(action_index=i,clock_s=now,deadline_s=expected,missed_slots=skip,requested_degrees=values,sent_degrees=sent,send_ms=cost))
   clock.wait_until(schedule.deadline,stop)
   emit('EPDONE',dict(ep=ep,clock_basis='simulation',actions_sent=len(rows),missed_action_ticks_after_startup=missed,actions=rows,published=published,mode=a.actuation_mode,started_clock_s=start,stopped_clock_s=clock.now(),duration_s=clock.now()-start))
 finally:restore();robot.disconnect()
if __name__=='__main__':main()
