"""Own a SIM-only deployment-realism window with immediate native targets."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time,sys
BASE=Path('/mnt/a4500-backup/exploring-vlas/eval');REPO=Path.home()/'Downloads/Exploring-VLAs'
sys.path.insert(0,str(REPO/'linux-env/scripts/eval'))
from phase3 import command,manifest,validate_checkpoint,validate_bank

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-by-utc',required=True);a=p.parse_args()
 root=a.out.resolve();assert root.is_relative_to(BASE/'runs') and not root.exists();root.mkdir()
 proof=validate_checkpoint(a.checkpoint);assert proof['model_sha256']==a.expected_sha256,'Checkpoint differs from the authorised weights';bank=BASE/'scene-bank-v1';banksha=validate_bank(bank)
 deadline=min(time.time()+74*60,datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-180)
 assert deadline>time.time()+600,'Insufficient time for safe startup/cleanup'
 env=dict(os.environ,SOARM_EVAL_LOCALHOST_ONLY='1',SOARM_COLLECTION_ROOT=str(BASE/'runtime33'),SOARM_INSTANCE_ROOT=str(BASE/'runtime33/instance92'),ISAACSIM_GPU='0',SOARM_POLICY_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds-eval.xml',PYTHONDONTWRITEBYTECODE='1',SOARM_EVAL_INFERENCE_MODE='eager',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for name in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(name,None)
 for name,sub in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),('TORCH_HOME','torch'),('TMPDIR','tmp'),('HF_HOME','huggingface'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','')]:
  directory=BASE/'cache'/sub;directory.mkdir(parents=True,exist_ok=True);env[name]=str(directory)
 source_hashes={name:hashlib.sha256((REPO/'linux-env/scripts/eval'/name).read_bytes()).hexdigest() for name in ('run_actuation_window.py','run_episodes.py','episode_client.py','actuation.py','sim_time.py','phase3.py','phase3_metrics.py','scene_reset.py')}
 ledger=dict(source_sha256_by_file=source_hashes,kind='deployment-realism, actuation v2',eval_actuation='immediate',action_duration_s=0.,started_utc=time.strftime('%FT%TZ',time.gmtime()),checkpoint=proof,bank_sha256=banksha,bank_condition='bank-v1 OOD cup arrangement',planned_seeds=list(range(900000,900030)),attempts=[],status='starting',root=str(root),deadline_unix=deadline,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 def save():
  tmp=root/'window-ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,root/'window-ledger.json')
 def run(cmd,logname,folder=None):
  with (root/logname).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['active_pid']=proc.pid;save();seen=set()
   try:
    while proc.poll() is None:
     if time.time()>deadline:raise TimeoutError('Announced window deadline')
     if folder:
      for path in sorted(folder.glob('ep*/result.json')):
       if path in seen:continue
       try:r=json.loads(path.read_text())
       except json.JSONDecodeError:continue
       seen.add(path);client=r.get('client',{});print(json.dumps(dict(event='episode',seed=r['seed'],color=r.get('color'),success=r.get('success'),grasp=r.get('grasp_success'),outcome=r['outcome'],missed=client.get('missed_action_ticks_after_startup'),empty_pct=r.get('empty_ticks_pct'),effective_hz=r.get('effective_action_fps'),actuation=r.get('eval_actuation'))),flush=True)
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
  index=0
  while index<30:
   rows=manifest()['per_checkpoint'][index:];folder=root/f'attempt-{len(ledger["attempts"])+1:02d}';receipt=root/(folder.name+'-manifest.json');receipt.write_text(json.dumps(rows,indent=2)+'\n')
   cmd=command(rows[0],a.checkpoint,folder,bank);cmd[cmd.index('--episodes')+1]=str(len(rows));cmd[cmd.index('--seeds')+1]=','.join(str(r['seed']) for r in rows);cmd+=['--episode-manifest',str(receipt),'--eval-actuation','immediate']
   attempt=dict(index=index,planned=len(rows),out=str(folder),command=cmd);ledger['attempts'].append(attempt);ledger['status']='scoring';save();rc=run(cmd,folder.name+'.log',folder)
   dirs=sorted(folder.glob('ep[0-9][0-9]'));attempt.update(exit_code=rc,attempted=len(dirs));save()
   assert dirs,'No episode receipts; inspect startup/warmup without retry'
   for directory in dirs:
    file=directory/'result.json';assert file.exists(),'Incomplete episode receipt; stop for audit'
    result=json.loads(file.read_text());assert result['eval_actuation']=='immediate' and result['actuation_duration_s']==0
    if result.get('client') and not result['client'].get('aborted'):assert result['client']['eval_actuation']=='immediate' and result['client']['actuation_duration_s']==0
   index+=len(dirs)
  ledger['status']='matrix-finished'
 except BaseException as error:
  ledger.update(status='stopped-with-error',error=f'{type(error).__name__}: {error}');raise
 finally:
  with (root/'cleanup.log').open('w') as log:stopped=subprocess.run(['bash','linux-env/scripts/collect_sim.sh','down','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
  ledger.update(cleanup_exit_code=stopped.returncode,stopped_utc=time.strftime('%FT%TZ',time.gmtime()));save();print(json.dumps(dict(event='GPU-window-stopped',status=ledger['status'],stopped_utc=ledger['stopped_utc'],cleanup_exit_code=stopped.returncode)),flush=True)
if __name__=='__main__':main()
