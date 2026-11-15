"""Own a native collector replay window in the private collector worktree."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time,shlex
from checkpoint_contract import require_handoff,require_processor_proof
REPO=Path(__file__).resolve().parents[3]
from native_environment import qualified_host,policy_python
def policy_plan(args):
 modes=(args.cadence_diagnostic,args.codec_diagnostic,args.replacement_diagnostic,args.memorisation_diagnostic,args.matched_reset_diagnostic)
 if sum(modes)>1:raise ValueError('Select only one native diagnostic')
 if args.matched_reset_diagnostic:
  settings=[('default',10,True,'weighted_average'),('whole50',50,False,'latest_only')]
  selection=getattr(args,'matched_reset_arms','both')
  if selection not in ('both','default','whole50'):raise ValueError('Unknown matched reset arm')
  if selection!='both':settings=[row for row in settings if row[0]==selection]
 elif args.memorisation_diagnostic:
  if args.best_replacement_cadence not in (10,25,50):raise ValueError('Memorisation requires a selected replacement cadence')
  settings=[('default',10,True,'weighted_average'),(f'best{args.best_replacement_cadence}',args.best_replacement_cadence,False,'latest_only')]
 elif args.replacement_diagnostic:settings=[(f'repl{n}',n,False,'latest_only') for n in (10,25,50)]
 elif args.cadence_diagnostic:settings=[(f'rtc-off-exec{n}',n,False,'weighted_average') for n in (1,5)]
 else:settings=[('attempt-01',10,True,'weighted_average')]
 return [dict(name=name,cadence=cadence,rtc=rtc,aggregation=aggregation,codec=args.codec_diagnostic) for name,cadence,rtc,aggregation in settings]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixture',type=Path);p.add_argument('--fixture-sha256');p.add_argument('--checkpoint',type=Path);p.add_argument('--weights-sha256');p.add_argument('--bank',type=Path);p.add_argument('--bank-sha256');p.add_argument('--cadence-diagnostic',action='store_true');p.add_argument('--stop-by-utc',required=True);p.add_argument('--codec-diagnostic',action='store_true');p.add_argument('--replacement-diagnostic',action='store_true');p.add_argument('--memorisation-diagnostic',action='store_true');p.add_argument('--matched-reset-diagnostic',action='store_true');p.add_argument('--matched-reset-qualification',action='store_true');p.add_argument('--matched-reset-arms',choices=('both','default','whole50'),default='both');p.add_argument('--bank-offset',type=int,default=0);p.add_argument('--bank-count',type=int);p.add_argument('--bootstrap-control',type=Path);p.add_argument('--reset-collector-root',type=Path);p.add_argument('--reset-qualification',type=Path);p.add_argument('--cpu-rpc-proof',type=Path);p.add_argument('--source-preflight-proof',type=Path);p.add_argument('--best-replacement-cadence',type=int,choices=(10,25,50));p.add_argument('--replay-commands',choices=('clean','executed','both'),default='both');p.add_argument('--capture-home-checks',action='store_true');p.add_argument('--policy-python',type=Path);p.add_argument('--processor-proof',type=Path);p.add_argument('--checkpoint-handoff',type=Path);p.add_argument('--replacement-selection',type=Path);a=p.parse_args();plan=policy_plan(a);guarded=a.cadence_diagnostic or a.codec_diagnostic or a.replacement_diagnostic or a.memorisation_diagnostic or a.matched_reset_diagnostic or a.matched_reset_qualification;host=qualified_host(REPO);HOST=Path(host['root']);interpreter=policy_python(host,a.policy_python) if a.checkpoint else None;root=a.out.resolve();assert root.is_relative_to(HOST/'eval/runs') and not root.exists()
 bank_rows=[];selected_rows=[];suite=None
 if a.checkpoint or a.matched_reset_qualification:
  from native_bank import load_bank,select_policy_rows
  bank_rows,suite=load_bank(a.bank,a.bank_sha256)
  if a.checkpoint:
   selected_rows=select_policy_rows(bank_rows,suite,a.bank_offset,a.bank_count)
   if suite=='heldout-bank-v2':assert a.matched_reset_diagnostic and a.matched_reset_arms!='both','Held-out scoring is one arm and one fixed15-seed chunk per window'
  else:
   assert a.bank_offset==0 and a.bank_count is None,'Qualification must visit the entire sealed bank'
   assert (suite=='heldout-bank-v2')==(a.bootstrap_control is not None),'Held-out qualification requires a seen bootstrap control'
 if a.matched_reset_arms!='both':assert a.matched_reset_diagnostic
 deadline=datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-180;assert deadline>time.time()+600
 if a.checkpoint:
  arms=[row['name'] for row in plan]
  assert all(len(str(root/name/'rpc.sock').encode())<104 for name in arms),'Native IPC path exceeds Unix socket limit'
  assert a.weights_sha256 and a.bank and a.bank_sha256
  assert hashlib.sha256((a.checkpoint/'model.safetensors').read_bytes()).hexdigest()==a.weights_sha256
 elif a.matched_reset_qualification:
  assert a.bank and a.bank_sha256 and a.reset_collector_root and not any((a.cadence_diagnostic,a.codec_diagnostic,a.replacement_diagnostic,a.memorisation_diagnostic,a.matched_reset_diagnostic))
 else:
  assert a.fixture and a.fixture_sha256
  assert hashlib.sha256(a.fixture.read_bytes()).hexdigest()==a.fixture_sha256
 assert not(a.matched_reset_qualification and a.checkpoint),'Reset qualification cannot also score a checkpoint'
 reset_qualification=None
 if a.matched_reset_qualification or a.matched_reset_diagnostic:
  from matched_reset import validate_collector_root
  assert a.reset_collector_root;validate_collector_root(a.reset_collector_root)
 bootstrap_control_proof=None
 if a.matched_reset_qualification and suite=='heldout-bank-v2':
  from native_bank import SEEN_BANK_SHA
  from matched_reset_client import require_qualification
  bootstrap_control_proof=require_qualification(a.bootstrap_control,host,a.reset_collector_root,SEEN_BANK_SHA,Path(__file__).parent,suite='seen-training-home',reference_sha256=bank_rows[0]['reference_run_sha256'])
 if a.matched_reset_diagnostic:
  from matched_reset_client import require_qualification
  assert a.reset_qualification
  reset_qualification=require_qualification(a.reset_qualification,host,a.reset_collector_root,a.bank_sha256,Path(__file__).parent,suite=suite,reference_sha256=bank_rows[0]['reference_run_sha256'])
  assert a.cpu_rpc_proof and a.source_preflight_proof
  cpu=json.loads(a.cpu_rpc_proof.read_text());preflight=json.loads(a.source_preflight_proof.read_text())
  assert cpu['status']=='PASS' and cpu['cuda_visible_devices']=='' and cpu['rtc_enabled'] and cpu['prefix_steps']==[0,40] and Path(cpu['checkpoint']).resolve()==a.checkpoint.resolve()
  assert preflight['status']=='PASS' and Path(preflight['checkpoint']).resolve()==a.checkpoint.resolve() and preflight['weights_sha256']==a.weights_sha256 and preflight['bank_sha256']==a.bank_sha256 and preflight['suite']==suite and preflight['seeds']==[r['seed'] for r in selected_rows]
  reset_qualification['cpu_rpc_proof']=dict(path=str(a.cpu_rpc_proof),sha256=hashlib.sha256(a.cpu_rpc_proof.read_bytes()).hexdigest())
  reset_qualification['source_preflight_proof']=dict(path=str(a.source_preflight_proof),sha256=hashlib.sha256(a.source_preflight_proof.read_bytes()).hexdigest())
 handoff=None;selection=None;processor_proof=None
 if a.memorisation_diagnostic or a.matched_reset_diagnostic:
  assert a.checkpoint_handoff
  if a.memorisation_diagnostic:assert a.replacement_selection
  if a.matched_reset_diagnostic:assert a.reset_collector_root
  handoff=require_handoff(a.checkpoint,a.checkpoint_handoff,a.weights_sha256,published=a.matched_reset_diagnostic)
  preprocessor=json.loads((a.checkpoint/'policy_preprocessor.json').read_text())
  if any('class' in step for step in preprocessor['steps']):
   assert a.processor_proof,'Custom saved processor requires actual checkpoint parity proof'
   processor_directory=json.loads(a.checkpoint_handoff.read_text())['source_before']['model'] if a.matched_reset_diagnostic else a.checkpoint
   processor_proof=require_processor_proof(processor_directory,a.processor_proof,handoff['files'],published_handoff=a.checkpoint_handoff)
  if a.memorisation_diagnostic:
   selection=json.loads(a.replacement_selection.read_text());assert selection['complete'] is True and selection['selected_cadence']==a.best_replacement_cadence
   assert len(selection['files_sha256'])==15
   assert hashlib.sha256((Path(selection['source_window'])/'window-ledger.json').read_bytes()).hexdigest()==selection['source_ledger_sha256']
   assert hashlib.sha256(Path(__file__).with_name('select_replacement.py').read_bytes()).hexdigest()==selection['selector_sha256']
   for path,sha in selection['files_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
   selection=dict(path=str(a.replacement_selection),sha256=hashlib.sha256(a.replacement_selection.read_bytes()).hexdigest(),decision=selection)
 LOOPBACK_SHA256='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'
 assert hashlib.sha256((REPO/'linux-env/cyclonedds.xml').read_bytes()).hexdigest()==LOOPBACK_SHA256
 instance=HOST/'eval/runtime/instance92';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SOARM_COLLECTION_ROOT=str(HOST),SOARM_INSTANCE_ROOT=str(instance),ISAACSIM_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=f'file://{REPO}/linux-env/cyclonedds.xml',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 if interpreter:env['SOARM_EVAL_POLICY_PYTHON']=interpreter
 host_source=(' && source '+shlex.quote(str(REPO/'linux-env/scripts/collect'/host['environment']))) if host.get('environment') else ''
 if guarded:env['SOARM_EVAL_TORCH_MEMORY_GIB']='3.0'
 for key in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(key,None)
 for key,part in [('TMPDIR','tmp'),('ROS_LOG_DIR','ros-logs'),('XDG_CACHE_HOME','cache'),('HF_HOME','cache/huggingface'),('OPTIX_CACHE_PATH','cache/optix')]:
  directory=instance/part;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
 interpreter_metadata=None
 if interpreter:
  code="import json,sys,torch;assert not torch.cuda.is_available() and not torch.cuda.is_initialized();print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized())))"
  interpreter_metadata=json.loads(subprocess.check_output([interpreter,'-c',code],env=dict(env,CUDA_VISIBLE_DEVICES=''),text=True,timeout=60))
 root.mkdir(parents=True)
 from native_gpu_audit import RegisteredGPUAudit
 registered_audit=RegisteredGPUAudit(instance,root/'gpu-ownership-registry.json') if guarded else None
 ledger=dict(bootstrap_control_proof=bootstrap_control_proof,suite=suite,selected_seeds=[r['seed'] for r in selected_rows],bank_offset=a.bank_offset,bank_count=len(selected_rows) if selected_rows else None,bootstrap_control=str(a.bootstrap_control) if a.bootstrap_control else None,bank=str(a.bank) if a.bank else None,bank_sha256=a.bank_sha256,reset_collector_root=str(a.reset_collector_root) if a.reset_collector_root else None,interpreter_metadata=interpreter_metadata,policy_python_sha256=hashlib.sha256(Path(interpreter).read_bytes()).hexdigest() if interpreter else None,reset_qualification=reset_qualification,host_profile=host,policy_python=interpreter,processor_proof=processor_proof,checkpoint_handoff=handoff,replacement_selection=selection,gpu_budget_gib=7 if guarded else None,global_memory_ceiling=.79 if guarded else None,started_utc=time.strftime('%FT%TZ',time.gmtime()),kind='UNSCORED matched collector reset qualification' if a.matched_reset_qualification else ('PRIMARY native held-tick heldout-bank-v2' if suite=='heldout-bank-v2' else 'DIAGNOSTIC matched collector reset seen-layout') if a.matched_reset_diagnostic else 'DIAGNOSTIC native memorisation default and replacement' if a.memorisation_diagnostic else 'DIAGNOSTIC native replacement-only RTC off' if a.replacement_diagnostic else 'DIAGNOSTIC native codec held-filler' if a.codec_diagnostic else 'DIAGNOSTIC native cadence RTC off' if a.cadence_diagnostic else 'PRIMARY native held-tick training-home policy test' if a.checkpoint else 'native held-tick actuation qualification',root=str(root),status='starting',arms=[],source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('run_native_replay_window.py','native_replay.py','held_tick.py','phase3_metrics.py','native_policy.py','native_policy_client.py','native_policy_robot.py','native_policy_server.py','codec_stream.py','checkpoint_contract.py','select_replacement.py','native_environment.py','native_gpu_audit.py','native_home_capture.py','matched_reset.py','matched_reset_client.py','native_bank.py','qualify_matched_reset.py')})
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
   from gpu_device import inventory
   from nvml_processes import sample
   try:
    cards=inventory();expected=next(c['uuid'] for c in cards if c['index']==0)
    audit,registration=registered_audit.sample(expected,cards,sample)
    owned=audit['per_gpu_memory_mib'][expected];entry.update(registration);entry['owned_mib']=owned;bad_samples[0]=0;ledger['peak_owned_mib']=max(ledger.get('peak_owned_mib',0),owned)
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
  if a.matched_reset_qualification:
   argv=['python3','-u',str(REPO/'linux-env/scripts/eval/qualify_matched_reset.py'),'--bank',str(a.bank),'--bank-sha256',a.bank_sha256,'--collector-root',str(a.reset_collector_root),'--out',str(root/'qualification')]+(['--bootstrap-control',str(a.bootstrap_control)] if a.bootstrap_control else [])
   shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+host_source+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
   ledger['status']='matched-reset-qualification';save();assert run(['bash','-c',shell],'qualification.log')==0,'Matched reset qualification failed; preserve evidence, no retry'
   proof=json.loads((root/'qualification/summary.json').read_text());assert proof['status']=='PASS' and proof['suite']==suite and proof['seeds']==[r['seed'] for r in bank_rows] and proof['episodes']==len(bank_rows) and proof['commands']==10*len(bank_rows)
   ledger['status']='matched-reset-qualified'
  elif a.checkpoint:
   for arm in plan:
    cadence=arm['cadence'];rtc=arm['rtc'];folder=root/arm['name'];argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_policy.py'),'--out',str(folder),'--checkpoint',str(a.checkpoint),'--weights-sha256',a.weights_sha256,'--bank',str(a.bank),'--bank-sha256',a.bank_sha256,'--observe-every-actions',str(cadence),'--queue-aggregation',arm['aggregation'],'--bank-offset',str(a.bank_offset)]+(['--bank-count',str(a.bank_count)] if a.bank_count is not None else [])+([] if rtc else ['--rtc-off'])
    if arm['codec']:argv.append('--eval-codec-roundtrip')
    if a.memorisation_diagnostic:argv.extend(['--evaluation-kind','DIAGNOSTIC memorisation seen-layout'])
    if a.matched_reset_diagnostic:argv.extend(['--collector-matched-reset','--reset-collector-root',str(a.reset_collector_root),'--evaluation-kind','PRIMARY native held-tick heldout-bank-v2' if suite=='heldout-bank-v2' else 'DIAGNOSTIC matched collector reset seen-layout'])
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+host_source+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
    ledger['status']='scoring-'+folder.name;ledger['arms'].append(dict(arm));save();rc=run(['bash','-c',shell],folder.name+'.log');assert rc==0,'Native policy window failed; preserve evidence, no retry'
    received=sorted(folder.glob('ep*/result.json'));assert len(received)==len(selected_rows)
    assert [json.loads(f.read_text())['seed'] for f in received]==[r['seed'] for r in selected_rows]
    for file in received:
     row=json.loads(file.read_text());print(json.dumps(dict(event='native-policy',cadence=cadence,rtc=rtc,seed=row['seed'],grasp=row.get('grasp_success'),success=row.get('success'),valid=row.get('valid'))),flush=True)
   ledger['status']='matrix-finished'
  else:
   for arm in (('clean','executed') if a.replay_commands=='both' else (a.replay_commands,)):
    folder=root/arm;argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_replay.py'),'--fixture',str(a.fixture),'--fixture-sha256',a.fixture_sha256,'--out',str(folder),'--commands',arm]
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+host_source+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
    record=dict(name=arm,started_utc=time.strftime('%FT%TZ',time.gmtime()));ledger['arms'].append(record);ledger['status']='replay-'+arm;save();rc=run(['bash','-c',shell],arm+'.log');record['exit_code']=rc;save();assert rc==0,'Native replay failed; preserve evidence, no retry'
    rows=json.loads((folder/'summary.json').read_text());assert len(rows)==3
    for row in rows:print(json.dumps(dict(event='native-replay',commands=arm,seed=row['seed'],grasp=row['grasp_success'],success=row['success'],preclosure_tcp_rms_mm=row['tcp_tracking']['preclosure_rms_mm'])),flush=True)
   if a.capture_home_checks:
    assert a.bank and a.bank_sha256
    argv=['python3','-u',str(REPO/'linux-env/scripts/eval/native_home_capture.py'),'--out',str(root/'home-cameras'),'--bank',str(a.bank),'--bank-sha256',a.bank_sha256]
    shell='source /opt/ros/humble/setup.bash && source '+shlex.quote(str(REPO/'vla_SO-ARM101/install/setup.bash'))+host_source+' && unset ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS && exec '+shlex.join(argv)
    assert run(['bash','-c',shell],'home-cameras.log')==0,'Native home/camera capture failed'
   ledger['status']='replay-finished'
 except BaseException as error:
  ledger.update(status='stopped-with-error',error=f'{type(error).__name__}: {error}');raise
 finally:
  with (root/'cleanup.log').open('w') as log:stopped=subprocess.run(['bash','linux-env/scripts/collect_sim.sh','down','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
  if registered_audit is not None:registered_audit.close()
  ledger.update(cleanup_exit_code=stopped.returncode,stopped_utc=time.strftime('%FT%TZ',time.gmtime()));save();print(json.dumps(dict(event='GPU-window-stopped',status=ledger['status'],stopped_utc=ledger['stopped_utc'],cleanup_exit_code=stopped.returncode)),flush=True)
if __name__=='__main__':main()
