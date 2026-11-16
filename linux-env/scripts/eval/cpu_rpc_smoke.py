"""CPU-only synthetic IPC -> production RobotClient/RPC -> real checkpoint smoke."""
from pathlib import Path
import argparse,json,os,secrets,signal,socket,subprocess,sys,threading,time
from multiprocessing.connection import Listener
import numpy as np
from native_environment import rpc_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--paired-flow-repeat',type=int);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--actions',type=int,default=20);p.add_argument('--observe-every-actions',type=int,choices=(1,5,10,25,50),default=10);p.add_argument('--rtc-off',action='store_true');p.add_argument('--queue-aggregation',choices=('weighted_average','latest_only'),default='weighted_average');p.add_argument('--eval-codec-roundtrip',action='store_true');a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 root=Path(__file__).resolve().parents[3];source_proof=rpc_sources(root);env=dict(os.environ,PYTHONPATH=str(Path(a.checkpoint).resolve())+os.pathsep+str(root/'lerobot/src'),SOARM_EVAL_NATIVE_HELD='1',LEROBOT_RTC_ENABLED='0' if a.rtc_off else '1',LEROBOT_RTC_EXECUTION_HORIZON='10',LEROBOT_RTC_MAX_GUIDANCE_WEIGHT='10',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for name in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCH_HOME'):
  if name not in env:env[name]=str(a.out/'cache'/name);Path(env[name]).mkdir(parents=True,exist_ok=True)
 key=a.out/'key';key.write_bytes(secrets.token_bytes(32));key.chmod(0o600);address=str(a.out/'ipc');assert len(address)<104
 listener=Listener(address,family='AF_UNIX',authkey=key.read_bytes());listener._listener._socket.settimeout(90);server=None;actor=None;connection=None;failure=[];done=threading.Event();state=dict(epoch=1,sim_ns=1_000_000_000,actions=0,captures=0);port=18102
 def bridge():
  nonlocal connection
  try:
   connection=listener.accept()
   while not done.is_set():
    if not connection.poll(.1):continue
    req=connection.recv()
    if req['op']=='capture':
     state['captures']+=1
     from native_policy_robot import FEATURES
     obs={n:float(i) for i,n in enumerate(FEATURES)};obs.update(top=np.zeros((480,640,3),np.uint8),wrist=np.full((480,640,3),127,np.uint8));connection.send(dict(observation=obs,metadata=dict(version=1,mode='collect',held=True,epoch=state['epoch'],sim_ns=state['sim_ns'])))
    elif req['op']=='status':connection.send(dict(epoch=state['epoch'],sim_ns=state['sim_ns']))
    elif req['op']=='action':state['actions']+=1;state['sim_ns']=1_000_000_000+round(state['actions']*1e9/30);connection.send(dict(performed=req['action']))
    else:raise RuntimeError('Unexpected operation')
  except EOFError:pass
  except BaseException as error:failure.append(str(error))
 def terminate(proc):
  if proc and proc.poll() is None:
   proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 try:
  with (a.out/'server.log').open('w') as log:
   server=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('native_policy_server.py')),'--host=127.0.0.1',f'--port={port}','--fps=30'],env=env,cwd=a.out,stdout=log,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+60
  while True:
   if server.poll() is not None:raise RuntimeError('CPU server startup failed')
   try:
    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
   except OSError:
    if time.monotonic()>deadline:raise TimeoutError('CPU server startup')
    time.sleep(.05)
  worker=threading.Thread(target=bridge,daemon=True);worker.start()
  with (a.out/'actor.log').open('w') as log:
   actor=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('native_policy_client.py')),'--socket',address,'--auth-file',str(key),'--calibration-dir',str(a.out/'calibration'),'--checkpoint',a.checkpoint,'--server-port',str(port),'--policy-device','cpu','--observe-every-actions',str(a.observe_every_actions),'--queue-aggregation',a.queue_aggregation]+(['--eval-codec-roundtrip'] if a.eval_codec_roundtrip else []),env=env,cwd=a.out,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
  # Input is buffered while the production checkpoint handshake completes.
  from flow_noise import episode_identity,validate_episode_noise
  identity=episode_identity(530000,a.paired_flow_repeat) if a.paired_flow_repeat is not None else None
  mode='RUNSEEDED' if identity is not None else 'RUN'
  payload=dict(task='Pick a red lego and place it in red cup',flow_noise=identity) if identity is not None else 'Pick a red lego and place it in red cup'
  actor.stdin.write(mode+' smoke '+str(a.actions)+' '+json.dumps(payload)+'\nQUIT\n');actor.stdin.flush();rc=actor.wait(timeout=600);done.set();worker.join(timeout=5);assert rc==0 and not failure,(rc,failure)
  text=(a.out/'actor.log').read_text()
  from held_tick import validate_policy_features,validate_policy_requests
  ready=json.loads(next(line[6:] for line in text.splitlines() if line.startswith('READY ')));validate_policy_features(ready['lerobot_features'])
  event=json.loads(next(line[7:] for line in text.splitlines() if line.startswith('EPDONE ')));assert state['actions']==a.actions and state['captures']==(a.actions-1)//a.observe_every_actions+1 and len(event['chunks'])==state['captures'] and event['actions_sent']==a.actions
  assert [c['native_rtc']['prefix_steps'] for c in event['chunks']]==([0]+[50-a.observe_every_actions]*(len(event['chunks'])-1) if not a.rtc_off else [0]*len(event['chunks']))
  assert all(len(c['actions_degrees'])==50 and all(len(x)==6 for x in c['actions_degrees']) for c in event['chunks'])
  validate_policy_requests(event['requests'],not a.rtc_off,a.observe_every_actions)
  config=json.loads((Path(a.checkpoint)/'config.json').read_text())
  validate_episode_noise(event,identity,config['chunk_size'],config['max_action_dim'])
  assert event['queue_aggregation']==a.queue_aggregation
  if a.queue_aggregation=='latest_only':
   from native_policy_robot import FEATURES
   for action in event['sent']:
    step=action['action_index'];chunk=event['chunks'][step//a.observe_every_actions]
    assert [action['requested'][n] for n in FEATURES]==chunk['actions_degrees'][step-chunk['step']]
  assert event['eval_codec_roundtrip']==a.eval_codec_roundtrip
  for request in event['requests']:
   if a.eval_codec_roundtrip:
    assert all(v['requested_codec_index']==v['returned_codec_index'] for v in request['capture']['eval_codec_roundtrip'].values())
   else:assert request['capture']['eval_codec_roundtrip'] is False
  assert rpc_sources(root)==source_proof,'RPC source changed during CPU smoke'
  proof=dict(flow_noise=identity,source_sha256=source_proof,queue_aggregation=a.queue_aggregation,eval_codec_roundtrip=a.eval_codec_roundtrip,status='PASS',scope='CPU synthetic transport only; no simulation or model quality claim',actions=a.actions,captures=state['captures'],observe_every_actions=a.observe_every_actions,rtc_enabled=not a.rtc_off,prefix_steps=[c['native_rtc']['prefix_steps'] for c in event['chunks']],checkpoint=a.checkpoint,inference_wait_wall_s=[r['wall_s'] for r in event['requests']],cuda_visible_devices=env['CUDA_VISIBLE_DEVICES']);(a.out/'result.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
 finally:
  done.set();terminate(actor);terminate(server)
  if connection is not None:connection.close()
  listener.close();key.unlink(missing_ok=True)
if __name__=='__main__':main()
