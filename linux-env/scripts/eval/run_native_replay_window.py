"""Own a native collector replay window in the private rtx6000 worktree."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time
REPO=Path(__file__).resolve().parents[3];HOST=Path('/mnt/storage3/aaugus11/exploring-vlas')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path);p.add_argument('--fixture-sha256');p.add_argument('--checkpoint',type=Path);p.add_argument('--weights-sha256');p.add_argument('--bank',type=Path);p.add_argument('--bank-sha256');p.add_argument('--stop-by-utc',required=True);a=p.parse_args();root=a.out.resolve();assert root.is_relative_to(HOST/'eval/runs') and not root.exists();root.mkdir(parents=True)
 deadline=datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-180;assert deadline>time.time()+600
 if a.checkpoint:
  assert a.weights_sha256 and a.bank and a.bank_sha256
  assert hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()==a.weights_sha256
 else:
  assert a.fixture and a.fixture_sha256
  assert hashlib.sha256(a.fixture.read_bytes()).hexdigest()==a.fixture_sha256
 LOOPBACK_SHA256='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'
 assert hashlib.sha256((REPO/'linux-env/cyclonedds.xml').read_bytes()).hexdigest()==LOOPBACK_SHA256
 instance=HOST/'eval/runtime/instance92';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SOARM_COLLECTION_ROOT=str(HOST),SOARM_INSTANCE_ROOT=str(instance),ISAACSIM_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds.xml',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for key in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(key,None)
 for key,part in [('TMPDIR','tmp'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','cache'),('HF_HOME','cache/huggingface'),('OPTIX_CACHE_PATH','cache/optix')]:
  directory=instance/part;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
 ledger=dict(started_utc=time.strftime('%FT%TZ',time.gmtime()),kind='PRIMARY native held-tick training-home policy test' if a.checkpoint else 'native held-tick actuation qualification',root=str(root),status='starting',arms=[],source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('run_native_replay_window.py','native_replay.py','held_tick.py','phase3_metrics.py','native_policy.py','native_policy_client.py','native_policy_robot.py','native_policy_server.py')})
 def save():
  tmp=root/'window-ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,root/'window-ledger.json')
 def run(cmd,logname):
  with (root/logname).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['active_pid']=proc.pid;save()
   try:
    while proc.poll() is None:
     if time.time()>deadline:raise TimeoutError('Native qualification deadline')
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
  assert run(['bash','linux-env/scripts/collect_sim.sh','up','2'],'launch.log')==0,'Private runtime launch failed'
  if a.checkpoint:
   folder=root/'attempt-01';argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_policy.py'),'--out',str(folder),'--checkpoint',str(a.checkpoint),'--weights-sha256',a.weights_sha256,'--bank',str(a.bank),'--bank-sha256',a.bank_sha256]
   import shlex
   shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+' && source '+shlex.quote(str(REPO/'linux-env/scripts/collect/hosts/rtx6000.env'))+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
   ledger['status']='scoring-training-home';save();rc=run(['bash','-c',shell],'policy.log');assert rc==0,'Native policy window failed; preserve evidence, no retry'
   received=sorted(folder.glob('ep*/result.json'));assert len(received)==5
   for file in received:
    row=json.loads(file.read_text());print(json.dumps(dict(event='native-policy',seed=row['seed'],grasp=row.get('grasp_success'),success=row.get('success'),valid=row.get('valid'))),flush=True)
   ledger['status']='matrix-finished'
  else:
   for arm in ('clean','executed'):
    folder=root/arm;argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_replay.py'),'--fixture',str(a.fixture),'--fixture-sha256',a.fixture_sha256,'--out',str(folder),'--commands',arm]
    import shlex
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+' && source '+shlex.quote(str(REPO/'linux-env/scripts/collect/hosts/rtx6000.env'))+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
    record=dict(name=arm,started_utc=time.strftime('%FT%TZ',time.gmtime()));ledger['arms'].append(record);ledger['status']='replay-'+arm;save();rc=run(['bash','-c',shell],arm+'.log');record['exit_code']=rc;save();assert rc==0,'Native replay failed; preserve evidence, no retry'
    rows=json.loads((folder/'summary.json').read_text());assert len(rows)==3
    for row in rows:print(json.dumps(dict(event='native-replay',commands=arm,seed=row['seed'],grasp=row['grasp_success'],success=row['success'],preclosure_tcp_rms_mm=row['tcp_tracking']['preclosure_rms_mm'])),flush=True)
   ledger['status']='replay-finished'
 except BaseException as error:
  ledger.update(status='stopped-with-error',error=f'{type(error).__name__}: {error}');raise
 finally:
  with (root/'cleanup.log').open('w') as log:stopped=subprocess.run(['bash','linux-env/scripts/collect_sim.sh','down','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
  ledger.update(cleanup_exit_code=stopped.returncode,stopped_utc=time.strftime('%FT%TZ',time.gmtime()));save();print(json.dumps(dict(event='GPU-window-stopped',status=ledger['status'],stopped_utc=ledger['stopped_utc'],cleanup_exit_code=stopped.returncode)),flush=True)
if __name__=='__main__':main()
