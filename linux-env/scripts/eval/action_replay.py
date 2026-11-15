"""Recorded actions through the native eval ROS backend; no policy inference."""
from pathlib import Path
import argparse,json,math,os,sys,time,queue,subprocess
import hashlib
from run_episodes import Ros,Client,mcp,ARM_JOINTS
from phase3 import new_score,require_deployment
from training_starts import entry,validate_bank
from rclpy.qos import qos_profile_sensor_data
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
REPO=Path.home()/'Downloads/Exploring-VLAs';BASE=Path('/mnt/a4500-backup/exploring-vlas/eval')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=Path,required=True);p.add_argument('--recorded-replay',type=Path,required=True);p.add_argument('--actuation-mode',choices=('native','immediate'),required=True);p.add_argument('--calibration',action='store_true');a=p.parse_args();root=a.out.resolve();assert root.is_relative_to(BASE/'runs') and not root.exists();root.mkdir()
 assert os.environ.get('ROS_DOMAIN_ID')=='92'
 os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp';os.environ['CYCLONEDDS_URI']=f'file://{REPO}/linux-env/cyclonedds-eval.xml'
 for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):os.environ.pop(name,None)
 plan=json.loads((a.starts/'home-plan.json').read_text());validate_bank(plan['bank'],plan['bank_sha256']);episodes=json.loads(a.recorded_replay.read_text());assert len(episodes)==(1 if a.calibration else 3)
 ros=Ros(True,'/joint_states_measured');deployment=require_deployment(ros,mcp);events=[];clock=lambda stamp:stamp.sec+stamp.nanosec/1e9
 def trajectory(label,message):events.append(dict(kind='received-command',controller=label,wall_s=time.time(),clock_s=ros.clock_s,joint_names=list(message.joint_names),positions_rad=list(message.points[0].positions),time_from_start_s=message.points[0].time_from_start.sec+message.points[0].time_from_start.nanosec/1e9))
 def controller(label,message):
  desired=getattr(message,'desired',getattr(message,'reference',None));actual=getattr(message,'actual',getattr(message,'feedback',None))
  events.append(dict(kind='controller-state',controller=label,wall_s=time.time(),clock_s=clock(message.header.stamp),joint_names=list(message.joint_names),desired_rad=list(desired.positions),actual_rad=list(actual.positions)))
 for label in ('arm','gripper'):
  ros.node.create_subscription(JointTrajectory,'/'+label+'_controller/joint_trajectory',lambda msg,label=label:trajectory(label,msg),qos_profile_sensor_data)
  ros.node.create_subscription(JointTrajectoryControllerState,'/'+label+'_controller/controller_state',lambda msg,label=label:controller(label,msg),qos_profile_sensor_data)
 ros.node.create_subscription(JointState,'/joint_states',lambda msg:events.append(dict(kind='mock-reference',wall_s=time.time(),clock_s=clock(msg.header.stamp),joint_names=list(msg.name),positions_rad=list(msg.position))),qos_profile_sensor_data)
 cmd=[str(Path.home()/'.pixi/bin/pixi'),'run','--manifest-path',str(REPO/'linux-env/pixi.toml'),'python','-u',str(Path(__file__).with_name('replay_client.py')),'--recorded-replay',str(a.recorded_replay),'--actuation-mode',a.actuation_mode,'--policy_type=smolvla','--pretrained_name_or_path=replay-no-model','--actions_per_chunk=50','--fps=30','--dispatch_fifo_priority=10','--dispatch_fifo_sudo=true','--robot.type=so101_ros2','--robot.actuate=true','--robot.use_degrees=true','--robot.joint_states_topic=/joint_states_measured','--robot.cameras={}','--robot.action_duration_s=0.03333333333333333']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for key in ('PYTHONPATH','AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH','LD_LIBRARY_PATH','ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION'):env.pop(key,None)
 client=None;recording=False
 try:
  client=Client(cmd,env,root/'client');ready=client.tagged('READY',90);(root/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
  for ep,record in episodes.items():
   out=root/ep;out.mkdir();state=entry(plan['bank'],record['seed'],plan['bank_sha256']);ros.home();ros.fjt(ros.arm,ARM_JOINTS,state['state'][:5],3);ros.fjt(ros.grip,['gripper_joint'],state['state'][5:],1)
   code=f'import sys;sys.path.insert(0,{str(Path(__file__).parent)!r});import scene_reset;result=scene_reset.prepare({record["seed"]},{plan["bank"]!r},("red","green","blue"),generate=False,training_start_sha256={plan["bank_sha256"]!r})'
   reset=mcp('run_python',{'code':code});cap=mcp('run_python',{'code':"s=carb.settings.get_settings();s.set_bool('/app/runLoops/main/syncToPresent',False);s.set_bool('/app/runLoops/main/rateLimitEnabled',True);s.set_float('/app/runLoops/main/rateLimitFrequency',30.);result={k:s.get(k) for k in ['/app/runLoops/main/rateLimitEnabled','/app/runLoops/main/rateLimitFrequency','/app/runLoops/main/syncToPresent']}"});assert cap['/app/runLoops/main/rateLimitFrequency']==30 and cap['/app/runLoops/main/rateLimitEnabled']
   deadline=time.monotonic()+15
   while ros.physical is None or ros.physical['epoch']!=reset['reset_epoch'] or ros.physical['sim_ns']<=reset['reset_stamp_ns']:
    if time.monotonic()>deadline:raise TimeoutError('Replay physical reset fence')
    time.sleep(.02)
   baseline=ros.physical;score=new_score(record['color'],baseline);events.clear();ros.physical_drain();initial=baseline
   mcp('soarm101.start_recording',{'file_name':str(out/'viewport_capture.mp4')});recording=True;client.send('RUN '+ep+' 999');start=client.tagged('EPSTART',15);score.begin=start['clock_s'];score.budget=len(record['actions_degrees'])/30+1
   physical=[initial];deadline=time.monotonic()+180;done=None
   while done is None:
    physical.extend(ros.physical_drain())
    try:done=client.tagged('EPDONE',.05)
    except queue.Empty:pass
    if time.monotonic()>deadline:raise TimeoutError('Recorded replay watchdog')
   time.sleep(.15);physical.extend(ros.physical_drain());video=mcp('soarm101.stop_recording');recording=False;unique={s['sim_ns']:s for s in physical};physical=[unique[k] for k in sorted(unique)]
   for snap in physical:
    stamp=snap['sim_ns']/1e9
    if start['clock_s']<=stamp<=done['stopped_clock_s']:score.update(snap)
   hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('action_replay.py','replay_client.py','actuation.py','scene_reset.py','training_starts.py','sim_time.py')}
   result=dict(seed=record['seed'],color=record['color'],mode=a.actuation_mode,start=start,client=done,reset=reset,cap=cap,video=video,source=record['source_frames'],source_sha256=record['source_frames_sha256'],code_sha256=hashes,**score.result())
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'physical.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in physical));(out/'controller-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));print(json.dumps(dict(seed=result['seed'],mode=a.actuation_mode,missed=done['missed_action_ticks_after_startup'],success=result['success'],grasp=result['grasp_success'])),flush=True)
 finally:
  if recording:
   try:mcp('soarm101.stop_recording')
   except Exception:pass
  if client is not None:client.close()
  ros.close()
if __name__=='__main__':main()
