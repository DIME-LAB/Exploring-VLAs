"""Own one announced, unscored execution-cadence window on six bank-v1 scenes."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
BASE=Path('/mnt/a4500-backup/exploring-vlas/eval');REPO=Path.home()/'Downloads/Exploring-VLAs'
sys.path.insert(0,str(REPO/'linux-env/scripts/eval'))
from phase3 import command,task,COLORS,validate_checkpoint,validate_bank

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-by-utc');a=p.parse_args();root=a.out.resolve()
 assert root.is_relative_to(BASE/'runs') and not root.exists(),'A diagnostic cannot overwrite a window';root.mkdir()
 model=BASE/'checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/030000/pretrained_model';bank=BASE/'scene-bank-v1'
 proof=validate_checkpoint(model);assert proof['model_sha256']=='87bc05bd7b2a8cab75febca69754dc23c16c116e7bf806bc5d6212b09847d31c';banksha=validate_bank(bank)
 start=time.time();deadline=start+74*60
 if a.stop_by_utc:deadline=min(deadline,datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-120)
 env=dict(os.environ,SOARM_EVAL_LOCALHOST_ONLY='1',SOARM_COLLECTION_ROOT=str(BASE/'runtime33'),SOARM_INSTANCE_ROOT=str(BASE/'runtime33/instance92'),ISAACSIM_GPU='0',SOARM_POLICY_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',PYTHONDONTWRITEBYTECODE='1',SOARM_EVAL_INFERENCE_MODE='eager',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(name,None)
 for name,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('TORCH_HOME','torch'),('TMPDIR','tmp'),('HF_HOME','huggingface'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','')]:
  directory=BASE/'cache'/sub;directory.mkdir(parents=True,exist_ok=True);env[name]=str(directory)
 ledger=dict(kind='DIAGNOSTIC, not frozen MAIN',started_utc=time.strftime('%FT%TZ',time.gmtime()),model=proof,bank_sha256=banksha,bank_condition='bank-v1 OOD cup arrangement',planned_episodes=24,seeds=list(range(900000,900006)),arms=[],status='starting',root=str(root),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),control='Separate traced 10-on repeat; original 30k MAIN receipts remain unchanged')
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
  for cadence,rtc in ((10,True),(25,True),(25,False),(50,False)):
   name=f'exec{cadence}-rtc-{"on" if rtc else "off"}';folder=root/name;rows=[dict(seed=seed,color=COLORS[seed%3],task=task(COLORS[seed%3]),rtc=rtc,suite='execution-diagnostic',phrase=f'exec{cadence}') for seed in range(900000,900006)]
   manifest=root/(name+'-manifest.json');manifest.write_text(json.dumps(rows,indent=2)+'\n')
   cmd=command(rows[0],model,folder,bank);cmd[cmd.index('--episodes')+1]='6';cmd[cmd.index('--seeds')+1]=','.join(str(r['seed']) for r in rows)
   cmd+=['--episode-manifest',str(manifest),'--diagnostic-reobserve-actions',str(cadence),'--diagnostic-prediction-trace']
   arm=dict(name=name,cadence=cadence,rtc=rtc,started_utc=time.strftime('%FT%TZ',time.gmtime()),command=cmd,expected_empty_retry_slots=None if cadence==50 else 10);ledger['arms'].append(arm);ledger['status']='running-'+name;save()
   print(json.dumps(dict(event='diagnostic-arm-start',**arm)),flush=True)
   rc=run(cmd,name+'.log');arm['exit_code']=rc;arm['finished_utc']=time.strftime('%FT%TZ',time.gmtime());save()
   received=sorted(folder.glob('ep*/result.json'));arm['received']=len(received);save()
   assert rc==0 and len(received)==6,'Diagnostic arm stopped; preserve all attempts, no automatic retry'
   for file in received:
    result=json.loads(file.read_text());client=result.get('client',{});assert client.get('observe_every_actions')==cadence
    assert result['rtc_enabled']==rtc and result['suite']=='execution-diagnostic'
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
