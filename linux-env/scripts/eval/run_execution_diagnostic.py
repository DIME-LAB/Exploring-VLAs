"""Own one announced execution-cadence or seen-training-start diagnostic."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time,shlex
BASE=Path('/mnt/a4500-backup/exploring-vlas/eval');REPO=Path.home()/'Downloads/Exploring-VLAs'
sys.path.insert(0,str(REPO/'linux-env/scripts/eval'))
from phase3 import command,task,COLORS,validate_checkpoint,validate_bank

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-by-utc');p.add_argument('--training-start-root',type=Path);p.add_argument('--replay-source',type=Path);p.add_argument('--actuation-mode',choices=('native','immediate'),default='native');p.add_argument('--calibration',action='store_true');a=p.parse_args();root=a.out.resolve()
 assert root.is_relative_to(BASE/'runs') and not root.exists(),'A diagnostic cannot overwrite a window';root.mkdir()
 model=BASE/'checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/030000/pretrained_model';bank=BASE/'scene-bank-v1'
 proof=None if a.replay_source else validate_checkpoint(model)
 if proof:assert proof['model_sha256']=='87bc05bd7b2a8cab75febca69754dc23c16c116e7bf806bc5d6212b09847d31c'
 arms=[]
 if a.replay_source:
  assert a.training_start_root
  from training_starts import validate_bank as validate_training_bank
  plan=json.loads((a.training_start_root/'home-plan.json').read_text());validate_training_bank(plan['bank'],plan['bank_sha256'])
  records=json.loads(a.replay_source.read_text());assert len(records)==(1 if a.calibration else 3)
  rows=[dict(seed=r['seed'],color=r['color'],task=r['task']) for r in records.values()]
  arms.append(('replay-'+a.actuation_mode,30,False,Path(plan['bank']),plan['bank_sha256'],rows,[]))
 elif a.training_start_root:
  from training_starts import validate_bank as validate_training_bank,SEEDS
  for mode in ('mid','home'):
   plan=json.loads((a.training_start_root/(mode+'-plan.json')).read_text());manifest=validate_training_bank(plan['bank'],plan['bank_sha256']);assert manifest['mode']==mode
   assert [r['seed'] for r in plan['rows']]==list(SEEDS) and all(r['suite']=='seen-training-'+mode and r['rtc'] is True for r in plan['rows'])
   arms.append((mode,10,True,Path(plan['bank']),plan['bank_sha256'],plan['rows'],['--training-start-bank-sha256',plan['bank_sha256']]))
 else:
  banksha=validate_bank(bank)
  for cadence,rtc in ((10,True),(25,True),(25,False),(50,False)):
   name=f'exec{cadence}-rtc-{"on" if rtc else "off"}';rows=[dict(seed=seed,color=COLORS[seed%3],task=task(COLORS[seed%3]),rtc=rtc,suite='execution-diagnostic',phrase=f'exec{cadence}') for seed in range(900000,900006)]
   arms.append((name,cadence,rtc,bank,banksha,rows,[]))
 start=time.time();deadline=start+74*60
 if a.stop_by_utc:deadline=min(deadline,datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-120)
 env=dict(os.environ,SOARM_EVAL_LOCALHOST_ONLY='1',SOARM_COLLECTION_ROOT=str(BASE/'runtime33'),SOARM_INSTANCE_ROOT=str(BASE/'runtime33/instance92'),ISAACSIM_GPU='0',SOARM_POLICY_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds-eval.xml',PYTHONDONTWRITEBYTECODE='1',SOARM_EVAL_INFERENCE_MODE='eager',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(name,None)
 for name,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('TORCH_HOME','torch'),('TMPDIR','tmp'),('HF_HOME','huggingface'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','')]:
  directory=BASE/'cache'/sub;directory.mkdir(parents=True,exist_ok=True);env[name]=str(directory)
 ledger=dict(kind='DIAGNOSTIC, not frozen MAIN',started_utc=time.strftime('%FT%TZ',time.gmtime()),model=proof,bank_sha256_by_arm={arm[0]:arm[4] for arm in arms},bank_condition='recorded-action replay' if a.replay_source else ('seen training layouts' if a.training_start_root else 'bank-v1 OOD cup arrangement'),planned_episodes=sum(len(arm[5]) for arm in arms),seeds=[r['seed'] for r in arms[0][5]],arms=[],status='starting',root=str(root),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),control='Exact recorded training actions; no model inference' if a.replay_source else ('Seen-layout mid and home starts; both hold measured state with zero velocities' if a.training_start_root else 'Separate traced 10-on repeat; original 30k MAIN receipts remain unchanged'))
 def save():
  tmp=root/'window-ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,root/'window-ledger.json')
 def run(cmd,logname):
  with (root/logname).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['active_pid']=proc.pid;save()
   try:
    while proc.poll() is None:
     if time.time()>deadline:raise TimeoutError('Diagnostic window deadline')
     time.sleep(.5)
    return proc.returncode
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGINT)
     try:proc.wait(timeout=35)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
    ledger.pop('active_pid',None);save()
 save()
 try:
  assert run(['bash','linux-env/scripts/collect_sim.sh','up','2'],'launch.log')==0,'Runtime launch failed'
  assert run(['bash','linux-env/scripts/collect_sim.sh','deploy-clock','2'],'deploy-clock.log')==0,'Deploy clock failed'
  for name,cadence,rtc,bank,banksha,rows,extra in arms:
   folder=root/name
   manifest=root/(name+'-manifest.json');manifest.write_text(json.dumps(rows,indent=2)+'\n')
   if a.replay_source:
    argv=['python3','-u',str(REPO/'linux-env/scripts/eval/action_replay.py'),'--out',str(folder),'--starts',str(a.training_start_root),'--recorded-replay',str(a.replay_source),'--actuation-mode',a.actuation_mode]+(['--calibration'] if a.calibration else [])
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+' && exec '+shlex.join(argv)
    cmd=['bash','-lc',shell]
   else:
    cmd=command(rows[0],model,folder,bank);cmd[cmd.index('--episodes')+1]=str(len(rows));cmd[cmd.index('--seeds')+1]=','.join(str(r['seed']) for r in rows)
    cmd+=['--episode-manifest',str(manifest),'--diagnostic-reobserve-actions',str(cadence),'--diagnostic-prediction-trace',*extra]
   arm=dict(name=name,cadence=cadence,rtc=rtc,started_utc=time.strftime('%FT%TZ',time.gmtime()),command=cmd,expected_empty_retry_slots=None if cadence==50 else 10);ledger['arms'].append(arm);ledger['status']='running-'+name;save()
   print(json.dumps(dict(event='diagnostic-arm-start',**arm)),flush=True)
   rc=run(cmd,name+'.log');arm['exit_code']=rc;arm['finished_utc']=time.strftime('%FT%TZ',time.gmtime());save()
   received=sorted(folder.glob('ep*/result.json'));arm['received']=len(received);save()
   assert rc==0 and len(received)==len(rows),'Diagnostic arm stopped; preserve all attempts, no automatic retry'
   for file in received:
    result=json.loads(file.read_text());client=result.get('client',{})
    if a.replay_source:
     assert len(client['published'])==2*client['actions_sent']
     expected_duration=0 if a.actuation_mode=='immediate' else 1/30
     assert all(abs(e['time_from_start_s']-expected_duration)<1e-8 for e in client['published'])
     print(json.dumps(dict(event='replay-episode',seed=result['seed'],mode=a.actuation_mode,missed=client['missed_action_ticks_after_startup'],grasp=result['grasp_success'],success=result['success'])),flush=True)
     continue
    assert client.get('observe_every_actions')==cadence
    assert result['rtc_enabled']==rtc and result['suite']==rows[0]['suite']
    chunks=client.get('chunk_events',[]);assert chunks,'No chunk telemetry'
    assert all(len(c['prediction_trace']['actions_degrees'])==50 and all(len(x)==6 for x in c['prediction_trace']['actions_degrees']) for c in chunks),'Incomplete chunk trace'
    assert client.get('empty_retry_slots')==arm['expected_empty_retry_slots']
    print(json.dumps(dict(event='diagnostic-episode',arm=name,seed=result['seed'],success=result.get('success'),grasp=result.get('grasp_success'),empty_pct=result.get('empty_ticks_pct'),missed=client.get('missed_action_ticks_after_startup'))),flush=True)
  ledger['status']='diagnostic-finished'
 except BaseException as error:
  ledger.update(status='stopped-with-error',error=f'{type(error).__name__}: {error}');raise
 finally:
  with (root/'cleanup.log').open('w') as log:
   stopped=subprocess.run(['bash','linux-env/scripts/collect_sim.sh','down','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
  ledger.update(cleanup_exit_code=stopped.returncode,stopped_utc=time.strftime('%FT%TZ',time.gmtime()));save();print(json.dumps(dict(event='GPU-window-stopped',**{k:ledger[k] for k in ('status','cleanup_exit_code','stopped_utc')})),flush=True)

if __name__=='__main__':main()
