"""Recorded actions through the actual production client and paced dispatcher."""
import argparse,json,os,sys,time,threading
from pathlib import Path
from queue import Queue
import draccus
import torch
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import TimedAction
from sim_time import install_client
from actuation import install

def emit(tag,data):print(tag+' '+json.dumps(data),flush=True)
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--recorded-replay',type=Path,required=True);p.add_argument('--actuation-mode',choices=('native','immediate'),default='native');a,remaining=p.parse_known_args();cfg=draccus.parse(RobotClientConfig,args=remaining)
 assert os.environ.get('ROS_DOMAIN_ID')=='92';cfg.robot.cameras={};cfg.observation_process=False
 client=RobotClient(cfg);install_client(client,True);client.start_barrier=threading.Barrier(1);clock=client.eval_clock;context={};published=[];rows=[];actions=[]
 def witness(label,msg):published.append(dict(context,publish_wall_s=time.time(),controller=label,joint_names=list(msg.joint_names),positions_rad=list(msg.points[0].positions),time_from_start_s=msg.points[0].time_from_start.sec+msg.points[0].time_from_start.nanosec/1e9,header_stamp_ns=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec))
 restore=install(client.robot,a.actuation_mode,True,witness)
 original=client.control_loop_action
 def traced_action(verbose):
  context.clear();context.update(action_index=client.latest_action+1,clock_s=clock.now())
  sent=original(verbose);step=client.latest_action
  rows.append(dict(action_index=step,clock_s=context['clock_s'],requested_degrees=actions[step],sent_degrees=sent,send_ms=client._last_dispatch_cost_s*1000))
  return sent
 client.control_loop_action=traced_action;client.control_loop_observation=lambda task,verbose,finish:None
 episodes=json.loads(a.recorded_replay.read_text());emit('READY',dict(backend=type(client.robot).__name__,dispatcher='actual RobotClient + sim_time.paced_control_loop',mode=a.actuation_mode,use_degrees=client.robot.config.use_degrees,actuate=client.robot.config.actuate,joint_names=client.robot.config.joint_names,requested_fifo=cfg.dispatch_fifo_priority))
 try:
  for line in sys.stdin:
   parts=line.split()
   if not parts:continue
   if parts[0] in ('QUIT','ABORT'):break
   if parts[0]!='RUN':raise ValueError('Unexpected replay command')
   ep=parts[1];record=episodes[ep];actions=record['actions_degrees'];published.clear();rows.clear();client.latest_action=-1;client.action_queue=Queue();client.shutdown_event.clear();start=clock.now()
   for i,values in enumerate(actions):client.action_queue.put(TimedAction(timestamp=time.time()+i/30,timestep=i,action=torch.tensor(values,dtype=torch.float32)))
   emit('EPSTART',dict(ep=ep,clock_s=start,actions=len(actions)))
   client.control_loop('',max_actions=len(actions),max_duration_s=len(actions)/30+5)
   clock.wait_until(rows[-1]['clock_s']+1/30,threading.Event());stats=client.control_loop_stats
   emit('EPDONE',dict(stats,ep=ep,actions=rows,published=published,mode=a.actuation_mode,started_clock_s=rows[0]['clock_s'],preparation_clock_s=start,stopped_clock_s=clock.now()))
 finally:restore();client.stop()
if __name__=='__main__':main()
