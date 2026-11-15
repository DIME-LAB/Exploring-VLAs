"""Own a native collector replay window in the private rtx6000 worktree."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time
from checkpoint_contract import require_handoff
REPO=Path(__file__).resolve().parents[3];HOST=Path('/mnt/storage3/aaugus11/exploring-vlas')
def policy_plan(args):
 modes=(args.cadence_diagnostic,args.codec_diagnostic,args.replacement_diagnostic,args.memorisation_diagnostic)
 if sum(modes)>1:raise ValueError('Select only one native diagnostic')
 if args.memorisation_diagnostic:
  if args.best_replacement_cadence not in (10,25,50):raise ValueError('Memorisation requires a selected replacement cadence')
  settings=[('default',10,True,'weighted_average'),(f'best{args.best_replacement_cadence}',args.best_replacement_cadence,False,'latest_only')]
 elif args.replacement_diagnostic:settings=[(f'repl{n}',n,False,'latest_only') for n in (10,25,50)]
 elif args.cadence_diagnostic:settings=[(f'rtc-off-exec{n}',n,False,'weighted_average') for n in (1,5)]
 else:settings=[('attempt-01',10,True,'weighted_average')]
 return [dict(name=name,cadence=cadence,rtc=rtc,aggregation=aggregation,codec=args.codec_diagnostic) for name,cadence,rtc,aggregation in settings]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path);p.add_argument('--fixture-sha256');p.add_argument('--checkpoint',type=Path);p.add_argument('--weights-sha256');p.add_argument('--bank',type=Path);p.add_argument('--bank-sha256');p.add_argument('--cadence-diagnostic',action='store_true');p.add_argument('--stop-by-utc',required=True);p.add_argument('--codec-diagnostic',action='store_true');p.add_argument('--replacement-diagnostic',action='store_true');p.add_argument('--memorisation-diagnostic',action='store_true');p.add_argument('--best-replacement-cadence',type=int,choices=(10,25,50));p.add_argument('--checkpoint-handoff',type=Path);p.add_argument('--replacement-selection',type=Path);a=p.parse_args();plan=policy_plan(a);guarded=a.cadence_diagnostic or a.codec_diagnostic or a.replacement_diagnostic or a.memorisation_diagnostic;root=a.out.resolve();assert root.is_relative_to(HOST/'eval/runs') and not root.exists();root.mkdir(parents=True)
 deadline=datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-180;assert deadline>time.time()+600
 if a.checkpoint:
  arms=[row['name'] for row in plan]
  assert all(len(str(root/name/'rpc.sock').encode())<104 for name in arms),'Native IPC path exceeds Unix socket limit'
  assert a.weights_sha256 and a.bank and a.bank_sha256
  assert hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()==a.weights_sha256
 else:
  assert a.fixture and a.fixture_sha256
  assert hashlib.sha256(a.fixture.read_bytes()).hexdigest()==a.fixture_sha256
 handoff=None;selection=None
 if a.memorisation_diagnostic:
  assert a.checkpoint_handoff and a.replacement_selection
  handoff=require_handoff(a.checkpoint,a.checkpoint_handoff,a.weights_sha256)
  selection=json.loads(a.replacement_selection.read_text());assert selection['complete'] is True and selection['selected_cadence']==a.best_replacement_cadence
  assert len(selection['files_sha256'])==15
  assert hashlib.sha256((Path(selection['source_window'])/'window-ledger.json').read_bytes()).hexdigest()==selection['source_ledger_sha256']
  assert hashlib.sha256(Path(__file__).with_name('select_replacement.py').read_bytes()).hexdigest()==selection['selector_sha256']
  for path,sha in selection['files_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
  selection=dict(path=str(a.replacement_selection),sha256=hashlib.sha256(a.replacement_selection.read_bytes()).hexdigest(),decision=selection)
 LOOPBACK_SHA256='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'
 assert hashlib.sha256((REPO/'linux-env/cyclonedds.xml').read_bytes()).hexdigest()==LOOPBACK_SHA256
 instance=HOST/'eval/runtime/instance92';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SOARM_COLLECTION_ROOT=str(HOST),SOARM_INSTANCE_ROOT=str(instance),ISAACSIM_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds.xml',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 if guarded:env['SOARM_EVAL_TORCH_MEMORY_GIB']='3.0'
 for key in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(key,None)
 for key,part in [('TMPDIR','tmp'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','cache'),('HF_HOME','cache/huggingface'),('OPTIX_CACHE_PATH','cache/optix')]:
  directory=instance/part;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
 ledger=dict(checkpoint_handoff=handoff,replacement_selection=selection,gpu_budget_gib=7 if guarded else None,global_memory_ceiling=.79 if guarded else None,started_utc=time.strftime('%FT%TZ',time.gmtime()),kind='DIAGNOSTIC native memorisation default and replacement' if a.memorisation_diagnostic else 'DIAGNOSTIC native replacement-only RTC off' if a.replacement_diagnostic else 'DIAGNOSTIC native codec held-filler' if a.codec_diagnostic else 'DIAGNOSTIC native cadence RTC off' if a.cadence_diagnostic else 'PRIMARY native held-tick training-home policy test' if a.checkpoint else 'native held-tick actuation qualification',root=str(root),status='starting',arms=[],source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('run_native_replay_window.py','native_replay.py','held_tick.py','phase3_metrics.py','native_policy.py','native_policy_client.py','native_policy_robot.py','native_policy_server.py','codec_stream.py','checkpoint_contract.py','select_replacement.py')})
 def save():
  tmp=root/'window-ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,root/'window-ledger.json')
 last_gpu_sample=[0.];bad_samples=[0]
 def memory_guard(logname):
  if not guarded or time.monotonic()-last_gpu_sample[0]<1:return
  last_gpu_sample[0]=time.monotonic()
  raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True);used,total=map(float,raw.strip().split(','));entry=dict(wall_s=time.time(),global_used_mib=used,total_mib=total,global_fraction=used/total)
  if logname!='launch.log':
   import sys
   sys.path.insert(0,str(REPO/'linux-env/scripts/collect'))
   from gpu_device import owned_snapshot,inventory,audit_sample,birth_set
   from nvml_processes import sample
   try:
    cards=inventory();expected=next(c['uuid'] for c in cards if c['index']==0);before=owned_snapshot(instance);observed=sample(cards);after=owned_snapshot(instance)
    assert birth_set(before)==birth_set(after),'Owned process births changed during GPU sample'
    audit=audit_sample(expected,before,cards,observed);assert audit['outcome']=='verified' and not audit['foreign_gpu_contexts'],audit
    owned=audit['per_gpu_memory_mib'][expected];entry['owned_mib']=owned;bad_samples[0]=0;ledger['peak_owned_mib']=max(ledger.get('peak_owned_mib',0),owned)
   except Exception as error:
    entry['audit_error']=str(error);bad_samples[0]+=1
  ledger['peak_global_fraction']=max(ledger.get('peak_global_fraction',0),entry['global_fraction'])
  with (root/'gpu-memory.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
  if entry.get('owned_mib',0)>7*1024:raise RuntimeError('Owned eval GPU use exceeded7GiB; stop only our processes')
  if used/total>=.79:raise RuntimeError('Shared device reached79% ceiling; release our eval allocation')
  if bad_samples[0]>=3:raise RuntimeError('Owned GPU attribution unverified for3samples')
 def run(cmd,logname):
  with (root/logname).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['active_pid']=proc.pid;save()
   try:
    while proc.poll() is None:
     if time.time()>deadline:raise TimeoutError('Native qualification deadline')
     memory_guard(logname)
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
   for arm in plan:
    cadence=arm['cadence'];rtc=arm['rtc'];folder=root/arm['name'];argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_policy.py'),'--out',str(folder),'--checkpoint',str(a.checkpoint),'--weights-sha256',a.weights_sha256,'--bank',str(a.bank),'--bank-sha256',a.bank_sha256,'--observe-every-actions',str(cadence),'--queue-aggregation',arm['aggregation']]+([] if rtc else ['--rtc-off'])
    if arm['codec']:argv.append('--eval-codec-roundtrip')
    if a.memorisation_diagnostic:argv.extend(['--evaluation-kind','DIAGNOSTIC memorisation seen-layout'])
    import shlex
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+' && source '+shlex.quote(str(REPO/'linux-env/scripts/collect/hosts/rtx6000.env'))+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
    ledger['status']='scoring-'+folder.name;ledger['arms'].append(dict(arm));save();rc=run(['bash','-c',shell],folder.name+'.log');assert rc==0,'Native policy window failed; preserve evidence, no retry'
    received=sorted(folder.glob('ep*/result.json'));assert len(received)==5
    for file in received:
     row=json.loads(file.read_text());print(json.dumps(dict(event='native-policy',cadence=cadence,rtc=rtc,seed=row['seed'],grasp=row.get('grasp_success'),success=row.get('success'),valid=row.get('valid'))),flush=True)
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
