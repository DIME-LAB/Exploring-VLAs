"""Production RobotClient/RPC/queue aggregation on collector-held action ticks."""
import argparse,json,os,sys,threading,time
from queue import Queue
from pathlib import Path
import torch
from native_policy_robot import HeldTickRobotConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2

def emit(tag,payload):print(tag+' '+json.dumps(payload),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--auth-file',required=True);p.add_argument('--calibration-dir',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--server-port',type=int,required=True);p.add_argument('--policy-device',choices=('cuda','cpu'),default='cuda');p.add_argument('--observe-every-actions',type=int,choices=(1,5,10,25,50),default=10);p.add_argument('--eval-codec-roundtrip',action='store_true');p.add_argument('--codec-audit-dir',type=Path);a=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized();torch.set_num_threads(4)
 cfg=RobotClientConfig(robot=HeldTickRobotConfig(socket=a.socket,auth_file=a.auth_file,calibration_dir=a.calibration_dir,eval_codec_roundtrip=a.eval_codec_roundtrip),policy_type='smolvla',pretrained_name_or_path=a.checkpoint,policy_device=a.policy_device,client_device='cpu',server_address=f'127.0.0.1:{a.server_port}',fps=30,actions_per_chunk=50,chunk_size_threshold=.8,observation_process=False,aggregate_fn_name='weighted_average')
 client=RobotClient(cfg);condition=threading.Condition();state=dict(paused=True,epoch_wall=float('inf'),chunks=[],latest_chunk=None);original=client._aggregate_action_queues
 def aggregate(incoming,aggregate_fn=None):
  with condition:
   if state['paused'] or not incoming or incoming[0].get_timestamp()<state['epoch_wall']:return
   original(incoming,aggregate_fn)
   event=dict(step=incoming[0].get_timestep(),arrival_wall_s=time.time(),native_rtc=getattr(incoming[0],'native_rtc',None),actions_degrees=[x.get_action().detach().cpu().tolist() for x in incoming])
   state['chunks'].append(event);state['latest_chunk']=event;condition.notify_all()
 client._aggregate_action_queues=aggregate
 if not client.start():client.stop();raise RuntimeError('Production policy handshake failed')
 client.start_barrier=threading.Barrier(1);receiver=threading.Thread(target=client.receive_actions,daemon=True);receiver.start();emit('READY',dict(lerobot_features=client.policy_config.lerobot_features))
 def request(task):
  step=client.latest_action+1;started=time.monotonic();client.control_loop_observation(task)
  capture=dict(client.robot.capture_metadata)
  with condition:
   if not condition.wait_for(lambda:state['latest_chunk'] is not None and state['latest_chunk']['step']==step,timeout=90):raise TimeoutError('Native policy chunk deadline')
   chunk=dict(state['latest_chunk'])
  status=client.robot.exchange(dict(op='status'))
  if status['sim_ns']!=capture['sim_ns'] or status['epoch']!=capture['epoch']:raise RuntimeError('Physics advanced during held inference')
  return dict(step=step,wall_s=time.monotonic()-started,measured_prediction_sim_s=(status['sim_ns']-capture['sim_ns'])/1e9,capture=capture,native_rtc=chunk['native_rtc'])
 try:
  for line in sys.stdin:
   parts=line.strip().split(maxsplit=3)
   if not parts:continue
   if parts[0]=='QUIT':break
   if parts[0] not in ('RUN','WARM'):raise ValueError('Expected RUN/WARM/QUIT')
   ep=parts[1];actions=int(parts[2]);task=json.loads(parts[3]);assert actions>0
   with condition:state.update(paused=True,epoch_wall=float('inf'),chunks=[],latest_chunk=None)
   client.stub.Ready(services_pb2.Empty())
   with client.action_queue_lock:client.action_queue=Queue()
   with client.latest_action_lock:client.latest_action=-1
   client.action_chunk_size=-1;client.must_go.set()
   with condition:state.update(paused=False,epoch_wall=time.time())
   assert Path(ep).name==ep and ep not in ('.','..')
   if a.codec_audit_dir is not None:
    assert a.eval_codec_roundtrip and a.codec_audit_dir.is_dir()
    client.robot.codec_audit_path=a.codec_audit_dir/(ep+'-first-encoded.npz')
   client.robot.reset_eval_codec()
   started=time.monotonic();requests=[];sent=[]
   emit('EPSTART',dict(ep=ep,warmup=parts[0]=='WARM'))
   for i in range(actions):
    if i%a.observe_every_actions==0:requests.append(request(task))
    performed=client.control_loop_action();sent.append(dict(action_index=i,performed=performed))
   with condition:state['paused']=True
   client.stub.Ready(services_pb2.Empty())
   emit('EPDONE',dict(ep=ep,actions_sent=len(sent),wall_s=time.monotonic()-started,requests=requests,chunks=state['chunks'],sent=sent,observe_every_actions=a.observe_every_actions,observation_trigger="fixed executed-action cadence; threshold not used as trigger",chunk_size_threshold=.8,actions_per_chunk=50,eval_codec_roundtrip=a.eval_codec_roundtrip,clock_basis='collector-native-held',inference_clock='simulation; physics waits during RPC'))
 finally:
  client.stop();receiver.join(timeout=5)
if __name__=='__main__':main()
