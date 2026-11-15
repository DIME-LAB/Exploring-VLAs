"""Own a scene-only golden resample and held-out bank publication window."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,time
REPO=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--reference-proof',type=Path,required=True);p.add_argument('--golden-destination',type=Path);p.add_argument('--stop-by-utc',required=True);a=p.parse_args()
    import sys
    sys.path.insert(0,str(REPO/'linux-env/scripts/collect'))
    from host_config import profile
    host=profile();base=Path(host['root']);root=a.out.resolve()
    assert host['name']=='dual-a4500' and root.is_relative_to(base/'eval/bank-generation') and not root.exists()
    qual=json.loads((a.qualification/'window-ledger.json').read_text());replays=json.loads((a.qualification/'executed/summary.json').read_text());homes=json.loads((a.qualification/'home-cameras/result.json').read_text())
    assert qual['cleanup_exit_code']==0 and qual['status']=='replay-finished'
    assert qual['host_profile']['name']==host['name'] and Path(qual['host_profile']['root'])==base and Path(qual['root']).resolve()==a.qualification.resolve()
    assert len(replays)==3 and all(r['success'] and r['grasp_success'] for r in replays)
    assert homes['status']=='PASS' and len(homes['rows'])==2
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='bf7336d93d5ba054c5edc66cea685725eb30eb40'
    for name,sha in {'collect.py':'09f86817b46e2b1ad4695fce29e059ea387a0bbc97d9568786a70c818308ae90','scene_bank.py':'eb41cef2e7bee6a972e646a26963c8a0d01142895223969c880144401e4ee868'}.items():assert hashlib.sha256((REPO/'linux-env/scripts/collect'/name).read_bytes()).hexdigest()==sha,'Reviewed scene sampler source drift'
    subprocess.run(['git','diff','--exit-code','HEAD','--','linux-env/scripts/collect',':(exclude)linux-env/scripts/collect/collect.py','isaac/exts','linux-env/scripts/collect_sim.sh','linux-env/scripts/isaac'],cwd=REPO,check=True)
    golden=(a.golden_destination or base/'eval/scene-bank-golden-v2').resolve();heldout=base/'eval/scene-bank-v2'
    assert golden.is_relative_to(base/'eval') and golden!=heldout and not golden.exists() and not heldout.exists()
    xml=REPO/'linux-env/cyclonedds.xml';assert hashlib.sha256(xml.read_bytes()).hexdigest()=='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'
    expected=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(line.startswith(expected) for line in apps.splitlines()),'GPU0 must be released before a new window'
    deadline=datetime.datetime.strptime(a.stop_by_utc,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc).timestamp()-180
    assert deadline>time.time()+600
    reference=base/'datasets/rtx6000-normal-i3-b00/normal';receipts=[reference/f'attempt-{i:04d}/receipt.json' for i in range(5)]
    assert all(file.is_file() for file in receipts) and (reference/'run.json').is_file()
    reference_proof=json.loads(a.reference_proof.read_text());assert reference_proof['passed']
    assert set(reference_proof['files'])=={'run.json',*[f'attempt-{i:04d}/receipt.json' for i in range(5)]}
    for name,sha in reference_proof['files'].items():assert hashlib.sha256((reference/name).read_bytes()).hexdigest()==sha
    assert [json.loads(file.read_text())['params']['seed'] for file in receipts]==list(range(530000,530005))
    root.mkdir(parents=True);instance=base/'eval/runtime/instance92'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SOARM_COLLECTION_ROOT=str(base),SOARM_INSTANCE_ROOT=str(instance),ISAACSIM_GPU='0',ROS_DOMAIN_ID='92',SOARM_KIT_MCP_PORT='8781',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI='file://'+str(xml),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'):env.pop(key,None)
    ledger=dict(started_utc=time.strftime('%FT%TZ',time.gmtime()),status='starting',expected_gpu=expected,reference_proof=reference_proof,reference_proof_sha256=hashlib.sha256(a.reference_proof.read_bytes()).hexdigest(),qualification_ledger_sha256=hashlib.sha256((a.qualification/'window-ledger.json').read_bytes()).hexdigest(),source_sha256={n:hashlib.sha256((REPO/'linux-env/scripts/collect'/n).read_bytes()).hexdigest() for n in ('collect.py','scene_bank.py','contract.py')},owner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),stages=[])
    def save():
        tmp=root/'window-ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,root/'window-ledger.json')
    def run(argv,name):
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['active_pid']=proc.pid;save()
            try:
                while proc.poll() is None:
                    if time.time()>deadline:raise TimeoutError('Scene bank window deadline')
                    time.sleep(.5)
                assert proc.returncode==0,f'{name} failed: preserve evidence; no retry'
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGINT)
                    try:proc.wait(timeout=35)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
                ledger.pop('active_pid',None);save()
    save()
    try:
        run(['bash','linux-env/scripts/collect_sim.sh','up','2'],'launch')
        for mode,name in [('receipts','golden-five'),('eval','heldout-thirty')]:
            if mode=='eval':
                # The sampler mutates the extension's CUP_LAYOUT dictionary.
                # A fresh Kit process restores the same startup baseline that
                # trained episodes used; no sampler reset or tolerance changes.
                ledger['status']='restart-before-heldout';save()
                run(['bash','linux-env/scripts/collect_sim.sh','down','2'],'golden-stop')
                run(['bash','linux-env/scripts/collect_sim.sh','up','2'],'heldout-launch')
            destination=golden if mode=='receipts' else heldout
            assert not destination.exists()
            argv=['bash','linux-env/scripts/collect_sim.sh','run','2','--scene-bank',mode,'--scene-bank-reference-run',str(reference/'run.json'),'--output',str(root/name),'--scene-bank-output',str(destination)]
            if mode=='receipts':argv+=['--scene-bank-receipts',*[str(f) for f in receipts]]
            ledger['status']=name;save();run(argv,name)
            proof=json.loads((root/name/'bank-complete.json').read_text());assert proof['complete'] and proof['layouts']==(5 if mode=='receipts' else 30)
            assert hashlib.sha256((destination/'bank-manifest.json').read_bytes()).hexdigest()==proof['manifest_sha256']
            ledger['stages'].append(proof);save();print(json.dumps(proof),flush=True)
        ledger['status']='banks-sealed'
    except BaseException as error:
        ledger.update(status='stopped-with-error',error=repr(error));raise
    finally:
        try:
            with (root/'cleanup.log').open('w') as log:rc=subprocess.run(['bash','linux-env/scripts/collect_sim.sh','down','2'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90).returncode
        except subprocess.TimeoutExpired:
            rc=124;ledger['cleanup_error']='Owned instance down timed out; inspect exact managed PIDs before further GPU work'
        ledger.update(cleanup_exit_code=rc,stopped_utc=time.strftime('%FT%TZ',time.gmtime()));save();print(json.dumps(dict(event='GPU-window-stopped',status=ledger['status'],stopped_utc=ledger['stopped_utc'],cleanup_exit_code=rc)),flush=True)
if __name__=='__main__':main()
