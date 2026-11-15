"""Exact training-layout home evaluation through collector ticks and production RPC."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,secrets,signal,socket,subprocess,sys,time
from multiprocessing.connection import Listener
import numpy as np
REPO=Path(__file__).resolve().parents[3];HOST=Path('/mnt/storage3/aaugus11/exploring-vlas')
sys.path.insert(0,str(REPO/'linux-env/scripts/collect'))
from collect import mcp
from held_tick import CollectorTransport,validate_policy_features,validate_policy_requests
from native_replay import require_environment,validate_collector_cut
from run_episodes import Client
from phase3_metrics import PhysicalScore
FEATURES=('shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--weights-sha256',required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--bank-sha256',required=True);p.add_argument('--server-port',type=int,default=18092);p.add_argument('--max-actions',type=int,default=1350);p.add_argument('--rtc-off',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--observe-every-actions',type=int,choices=(1,5,10,25,50),default=10);p.add_argument('--queue-aggregation',choices=('weighted_average','latest_only'),default='weighted_average');p.add_argument('--eval-codec-roundtrip',action='store_true');a=p.parse_args();require_environment();core=validate_collector_cut()
 assert a.out.resolve().is_relative_to(HOST/'eval/runs') and not a.out.exists();a.out.mkdir();assert a.max_actions==1350
 assert hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()==a.weights_sha256
 checkpoint_config=json.loads((a.checkpoint/'config.json').read_text());assert [k for k in checkpoint_config['input_features'] if k.startswith('observation.images.')]==['observation.images.top','observation.images.wrist']
 assert all(checkpoint_config['input_features'][k]['shape']==[3,480,640] for k in ('observation.images.top','observation.images.wrist'))
 assert checkpoint_config['input_features']['observation.state']['shape']==[6] and checkpoint_config['output_features']['action']['shape']==[6] and checkpoint_config['chunk_size']==50
 bankfile=a.bank/'bank-manifest.json';assert hashlib.sha256(bankfile.read_bytes()).hexdigest()==a.bank_sha256;bank=json.loads(bankfile.read_text());assert bank['kind']=='seen-training-start-diagnostic' and bank['mode']=='home' and bank['seeds']==list(range(530000,530005))
 for name,sha in bank['files_sha256'].items():assert hashlib.sha256((a.bank/name).read_bytes()).hexdigest()==sha
 rows=[]
 for seed in bank['seeds']:
  state=json.loads((a.bank/f'state-{seed}.json').read_text());bodies=json.loads((a.bank/f'seed-{seed}.json').read_text());rows.append(dict(state,bodies=bodies))
 if a.check_only:
  proof=dict(status='PASS',scope='Native policy source/model/bank/environment preflight; no GPU or simulation',weights_sha256=a.weights_sha256,bank_sha256=a.bank_sha256,seeds=[r['seed'] for r in rows],collector_source_sha256=core)
  (a.out/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True);return
 fkpath=REPO/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py';spec=importlib.util.spec_from_file_location('native_policy_fk',fkpath);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
 limits_path=REPO/'vla_SO-ARM101/src/so_arm101_control';sys.path.insert(0,str(limits_path));from so_arm101_control.grasp_geometry import JOINT_LIMITS
 names=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper_joint');bounds=np.asarray([JOINT_LIMITS[n] for n in names])
 transport=None;listener=None;connection=None;actor=None;server=None;serverlog=None;recording=False
 env={k:v for k,v in os.environ.items() if not k.startswith(('ROS_','AMENT_','COLCON_','CMAKE_','PYTHON')) and k!='LD_LIBRARY_PATH'}
 env.update(PYTHONPATH=str(REPO/'lerobot/src'),PYTHONDONTWRITEBYTECODE='1',SOARM_EVAL_NATIVE_HELD='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LEROBOT_RTC_ENABLED='0' if a.rtc_off else '1',LEROBOT_RTC_EXECUTION_HORIZON='10',LEROBOT_RTC_MAX_GUIDANCE_WEIGHT='10',LEROBOT_RTC_PREFIX_ATTENTION_SCHEDULE='EXP')
 for key,part in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('HF_HOME','cache/huggingface'),('TORCH_HOME','cache/torch'),('TORCHINDUCTOR_CACHE_DIR','cache/inductor'),('TRITON_CACHE_DIR','cache/triton')]:
  directory=HOST/'eval'/part;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
 python='/mnt/storage3/aaugus11/smolvla-train/venv/bin/python';key=a.out/'rpc.key';fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as file:file.write(secrets.token_bytes(32))
 address=str(a.out/'rpc.sock');assert len(address.encode())<104
 config=dict(queue_aggregation=a.queue_aggregation,eval_codec_roundtrip=a.eval_codec_roundtrip,codec_context='causal running per-camera codec with held-image filler; diagnostic temporal approximation' if a.eval_codec_roundtrip else None,model=str(a.checkpoint),weights_sha256=a.weights_sha256,bank=str(a.bank),bank_sha256=a.bank_sha256,collector_source_sha256=core,rtc=not a.rtc_off,actions_per_chunk=50,observe_every_actions=a.observe_every_actions,chunk_size_threshold=.8,execution_horizon=10,max_guidance_weight=10,inference_mode='eager',clock='collector held ticks; physics waits during inference',state='measured native radians converted to degrees',cameras=['top','wrist'],fork_commit=subprocess.check_output(['git','-C',str(REPO/'lerobot'),'rev-parse','HEAD'],text=True).strip(),host_load_start=list(os.getloadavg()))
 assert config['fork_commit']=='e1e201098b2056827908e76f58bbf5733beb4d82'
 subprocess.run(['git','-C',str(REPO/'lerobot'),'diff','--quiet','HEAD'],check=True)
 (a.out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
 def terminate(proc):
  if proc is not None and proc.poll() is None:
   proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
 try:
  transport=CollectorTransport();mcp('run_python',{'code':(REPO/'linux-env/scripts/collect/configure_clock.py').read_text()+'\nresult=configure(30)'})
  installed=mcp('run_python',{'code':f'import sys\nsys.path.insert(0,{str(REPO/"linux-env/scripts/collect")!r})\nimport collection_runtime as c\nresult=c.install("collect")'});assert installed['loaded_sha256']==core['collection_runtime.py'] and installed['articulation_audit']['fixed_base'] is True and installed['physics_config']['dynamics']=='cpu';transport.generation.install(installed)
  assert installed['cuda_device_count']==1 and installed['cuda_device']==0 and installed['renderer_multi_gpu'] is False
  (a.out/'native-runtime.json').write_text(json.dumps(installed,indent=2)+'\n')
  deadline=time.monotonic()+20
  while transport.publisher.get_subscription_count()<1:
   if time.monotonic()>deadline:raise TimeoutError('Native command graph discovery')
   time.sleep(.05)
  mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read()
  with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as portcheck:portcheck.bind(('127.0.0.1',a.server_port))
  server_env=dict(env,CUDA_VISIBLE_DEVICES=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip());serverlog=(a.out/'server.log').open('w');server=subprocess.Popen([python,'-u',str(Path(__file__).with_name('native_policy_server.py')),'--host=127.0.0.1',f'--port={a.server_port}','--fps=30'],env=server_env,cwd=a.out,stdout=serverlog,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+90
  while True:
   if server.poll() is not None:raise RuntimeError('Native policy server exited')
   try:
    with socket.create_connection(('127.0.0.1',a.server_port),timeout=.2):break
   except OSError:
    if time.monotonic()>deadline:raise TimeoutError('Native policy server startup')
    time.sleep(.05)
  listener=Listener(address,family='AF_UNIX',authkey=key.read_bytes());os.chmod(address,0o600);listener._listener._socket.settimeout(90)
  actor_env=dict(env,CUDA_VISIBLE_DEVICES='');cmd=[python,'-u',str(Path(__file__).with_name('native_policy_client.py')),'--socket',address,'--auth-file',str(key),'--calibration-dir',str(HOST/'eval/cache/calibration'), '--checkpoint',str(a.checkpoint),'--server-port',str(a.server_port),'--observe-every-actions',str(a.observe_every_actions),'--queue-aggregation',a.queue_aggregation]
  if a.eval_codec_roundtrip:cmd.extend(['--eval-codec-roundtrip','--codec-audit-dir',str(a.out)])
  actor=Client(cmd,actor_env,a.out/'client');connection=listener.accept();ready=actor.tagged('READY',300);(a.out/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
  validate_policy_features(ready['lerobot_features'])
  active={}
  def native_identity():
   identity=mcp('run_python',{'code':'import collection_runtime as c\nr=c.runtime()\nresult=dict(mode=r.mode,held=not r.timeline.is_playing(),busy=r.busy,sim_ns=r.stamp(),epoch=r.epoch,failure=r.failure)'})
   current=transport.session.current
   assert identity['mode']=='collect' and identity['held'] and not identity['busy'] and identity['failure'] is None
   assert identity['epoch']==current['epoch'] and abs(identity['sim_ns']-current['sim_ns'])<=1000
   return identity
  def run_actor(ep,task,actions,warm=False):
   baseline=transport.session.current;snapshots=[baseline];commands=[];captures=[];active.clear();active.update(physical=snapshots,commands=commands,captures=captures,first_images={});scorer=PhysicalScore(color,baseline,fk);scorer.update(baseline);hold=list(baseline['state']);started=time.monotonic();actor.send(('WARM' if warm else 'RUN')+' '+ep+' '+str(actions)+' '+json.dumps(task));actor.tagged('EPSTART',30)
   while True:
    if server.poll() is not None:raise RuntimeError('Policy server exited during episode')
    if actor.proc.poll() is not None:raise RuntimeError('Policy client exited during episode')
    if time.monotonic()-started>900:raise TimeoutError('Native episode wall watchdog')
    if connection.poll(.02):
     request=connection.recv()
     try:
      current=transport.session.current
      if request['op']=='capture':
       identity=native_identity()
       observation={name:math.degrees(q) for name,q in zip(FEATURES,current['state'])};images={camera:transport.sensors.image(transport.session.entry[camera][0]).copy() for camera in ('top','wrist')};observation.update(images)
       if not active['first_images']:active['first_images']=images
       metadata=dict(version=1,mode='collect',held=True,sim_ns=current['sim_ns'],epoch=current['epoch'],state_degrees=[observation[n] for n in FEATURES],camera_rgb_sha256={camera:hashlib.sha256(image.tobytes()).hexdigest() for camera,image in images.items()},camera_identity={camera:transport.session.entry[camera][2] for camera in images});captures.append(metadata);connection.send(dict(observation=observation,metadata=metadata))
      elif request['op']=='status':connection.send(native_identity())
      elif request['op']=='action':
       values=np.asarray([request['action'][n] for n in FEATURES]);assert np.all(np.isfinite(values));rad=np.radians(values);command=hold if warm else np.clip(rad,bounds[:,0],bounds[:,1]).tolist();proof,current=transport.session.step(command);commands.append(dict(action_index=len(commands),requested_degrees=values.tolist(),warmup_hold=warm,**proof));snapshots.append(current);scorer.update(current);connection.send(dict(performed={name:math.degrees(q) for name,q in zip(FEATURES,proof['ack']['applied_command'])}))
      else:raise ValueError('Unknown native IPC operation')
     except BaseException as error:
      try:connection.send(dict(error=str(error)))
      except Exception:pass
      raise
    try:done=actor.tagged('EPDONE',.001);break
    except __import__('queue').Empty:pass
   assert len(commands)==actions==done['actions_sent']
   validate_policy_requests(done['requests'],not a.rtc_off,a.observe_every_actions)
   assert done['queue_aggregation']==a.queue_aggregation
   if a.queue_aggregation=='latest_only':
    for action in done['sent']:
     step=action['action_index'];chunk=done['chunks'][step//a.observe_every_actions]
     assert step-chunk['step']>=0 and step-chunk['step']<50
     assert [action['requested'][n] for n in FEATURES]==chunk['actions_degrees'][step-chunk['step']],'Replacement queue modified the latest raw action'
   assert done['eval_codec_roundtrip']==a.eval_codec_roundtrip
   for request in done['requests']:
    codec=request['capture']['eval_codec_roundtrip']
    if a.eval_codec_roundtrip:
     assert set(codec)=={'top','wrist'} and all(v['requested_codec_index']==v['returned_codec_index'] and v['mode']=='running-two-stage-held-lookahead-diagnostic' for v in codec.values())
    else:assert codec is False
   return done,snapshots,commands,captures,scorer.result()
  # Warm predictions while only the initial native home target is actuated.
  color=rows[0]['color'];done,*_=run_actor('warmup',rows[0]['task'],100,True);(a.out/'warmup.json').write_text(json.dumps(done,indent=2)+'\n')
  for index,row in enumerate(rows):
   out=a.out/f'ep{index:02d}';out.mkdir();color=row['color'];started=time.monotonic();load=list(os.getloadavg());reset=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().restore('+repr(dict(state=row['state'],drive_targets=row['recorded_drive_targets'],bodies=row['bodies']))+',arm=True,restore_drives=True)'});transport.generation.install(reset);transport.sensors.prune_durable_camera_history();mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read()
   actual_base=transport.session.current['robot_base_world_pose'];expected_base=row['robot_base_world_pose']
   assert max(abs(x-y) for x,y in zip(actual_base['position'],expected_base['position']))<=1e-6
   assert min(max(abs(x-y) for x,y in zip(actual_base['orientation'],expected_base['orientation'])),max(abs(x+y) for x,y in zip(actual_base['orientation'],expected_base['orientation'])))<=1e-6
   mcp('soarm101.start_recording',{'file_name':str(out/'viewport_capture.mp4')});recording=True
   try:
    done,snapshots,commands,captures,score=run_actor(out.name,row['task'],a.max_actions)
   except BaseException as error:
    (out/'result.json').write_text(json.dumps(dict(seed=row['seed'],color=color,task=row['task'],valid=False,outcome='invalid',error=f'{type(error).__name__}: {error}',kind=f'DIAGNOSTIC replacement-only cadence{a.observe_every_actions}' if a.queue_aggregation=='latest_only' else 'DIAGNOSTIC causal codec held-lookahead' if a.eval_codec_roundtrip else f'DIAGNOSTIC native held-tick RTC-off cadence{a.observe_every_actions}' if a.rtc_off or a.observe_every_actions!=10 else 'PRIMARY native held-tick policy evaluation',weights_sha256=a.weights_sha256,bank_sha256=a.bank_sha256),indent=2)+'\n')
    for key,filename in (('physical','physical.jsonl'),('commands','commands.jsonl'),('captures','captures.jsonl')):(out/filename).write_text(''.join(json.dumps(v)+'\n' for v in active.get(key,[])))
    raise
   video=mcp('soarm101.stop_recording');recording=False
   stamps=np.asarray([s['sim_ns'] for s in snapshots]);jitter=(stamps[:-1]-stamps[0]-np.arange(len(commands))*1e9/30)/1e6
   result=dict(queue_aggregation=a.queue_aggregation,eval_codec_roundtrip=a.eval_codec_roundtrip,seed=row['seed'],color=color,task=row['task'],kind=f'DIAGNOSTIC replacement-only cadence{a.observe_every_actions}' if a.queue_aggregation=='latest_only' else 'DIAGNOSTIC causal codec held-lookahead' if a.eval_codec_roundtrip else f'DIAGNOSTIC native held-tick RTC-off cadence{a.observe_every_actions}' if a.rtc_off or a.observe_every_actions!=10 else 'PRIMARY native held-tick policy evaluation',suite='seen-training-home',**score,client=done,reset=reset,video=video,requested_fps=30,effective_action_fps=len(commands)/((stamps[-1]-stamps[0])/1e9),sim_time_s=(stamps[-1]-stamps[0])/1e9,wall_time_s=time.monotonic()-started,sim_dispatch_jitter_p95_ms=float(np.quantile(np.abs(jitter),.95)),missed_sim_slots=0,empty_sim_ticks=0,empty_sim_tick_semantics='Physics waits while inference runs; report wall inference waits separately',host_load_start=load,host_load_end=list(os.getloadavg()),bank_sha256=a.bank_sha256,weights_sha256=a.weights_sha256,rtc_enabled=not a.rtc_off,actuation='native held tick',valid=True)
   np.savez_compressed(out/'first-cameras.npz',**active['first_images'])
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'physical.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in snapshots));(out/'commands.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in commands));(out/'captures.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in captures));print(json.dumps(dict(event='native-policy',seed=row['seed'],color=color,success=result['success'],grasp=result['grasp_success'],closest_m=result['closest_target_distance_m'])),flush=True)
 finally:
  if recording:
   try:mcp('soarm101.stop_recording')
   except Exception:pass
  cleanup=[]
  for label,callback in [('actor',lambda:actor.close() if actor is not None else None),('server',lambda:terminate(server)),('serverlog',lambda:serverlog.close() if serverlog is not None else None),('connection',lambda:connection.close() if connection is not None else None),('listener',lambda:listener.close() if listener is not None else None),('transport',lambda:transport.close() if transport is not None else None)]:
   try:callback()
   except BaseException as error:cleanup.append(dict(component=label,error=str(error)))
  key.unlink(missing_ok=True)
  (a.out/'process-cleanup.json').write_text(json.dumps(dict(errors=cleanup),indent=2)+'\n')
if __name__=='__main__':main()
