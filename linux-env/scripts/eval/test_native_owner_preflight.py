"""Run the actual owner CLI in an isolated CPU interpreter with a fake Kit."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,unittest

def exercise(xml,scenario="pass"):
    from unittest import mock
    from types import SimpleNamespace
    import datetime,hashlib,time
    folder=Path(__file__).resolve().parent
    sys.path.insert(0,str(folder))
    import run_native_replay_window as owner
    import native_bank,matched_reset
    import native_gpu_audit
    import gpu_device,nvml_processes
    from managed_process import start,stop
    from run_smoke import identity
    with tempfile.TemporaryDirectory() as d:
        base=Path(d);repo=base/'repo';host=base/'host';out=host/'eval/runs/preflight'
        (repo/'linux-env').mkdir(parents=True)
        (repo/'linux-env/cyclonedds.xml').write_bytes(Path(xml).read_bytes())
        instance=host/'eval/runtime/instance92';instance.mkdir(parents=True)
        paths=[];calls=[];registered=[];tick=[0.];external_claim=None
        seeds=list(native_bank.SEEN15_SEEDS) if scenario=='seen15' else list(range(530000,530005))
        suite='seen-training-home-15' if scenario=='seen15' else 'seen-training-home'
        cards=[dict(uuid='gpu0',index=0),dict(uuid='gpu1',index=1)]
        external=os.getppid()
        def nvml(_):
            rows=[dict(pid=external,gpu_uuid='gpu0',api='compute' if scenario=='prelaunch-compute' else 'graphics',memory_bytes=4096)]
            if paths:
                rows.append(dict(pid=os.getpid(),gpu_uuid='gpu0',api='compute',memory_bytes=3700*1024**2))
                if scenario in ('late-graphics','late-benign-graphics'):rows.append(dict(pid=1,gpu_uuid='gpu0',api='graphics',memory_bytes=(17 if scenario=='late-graphics' else 3)*1024**2))
            return dict(rows=rows,coverage={c['uuid']:dict(compute='verified',graphics='verified') for c in cards},errors=[])
        class FakeKit:
            def __init__(self,cmd,**kw):
                self.pid=os.getpid();self.returncode=None;self.polls=0
                calls.append(cmd)
                if cmd[:2]==['bash','linux-env/scripts/collect_sim.sh']:
                    assert (out/'external-graphics-baseline.json').exists(),'Baseline must precede runtime launch'
                    fd=int(kw['env']['SOARM_INSTANCE_CLAIM_FD']);assert kw['pass_fds']==(fd,)
                    helper=folder.parent/'collect/instance_claim.sh'
                    subprocess.run(['bash','-c','source "$1"; claim_collection_instance "$2"','bash',str(helper),str(instance)],env=kw['env'],pass_fds=(fd,),check=True)
                    from run_smoke import InstanceClaim
                    try:other=InstanceClaim(instance)
                    except RuntimeError:pass
                    else:other.close();raise AssertionError('Owner did not hold exclusive instance claim')
                    for name in ('isaacsim-soarm101.pid','control.pid'):
                        path=instance/name
                        start(path,instance/(name+'.log'),[sys.executable,'-c','import time;time.sleep(120)']);paths.append(path)
                else:
                    assert 'qualify_matched_reset.py' in cmd[-1]
                    q=out/'qualification';q.mkdir()
                    (q/'summary.json').write_text(json.dumps(dict(status='PASS',suite=suite,seeds=seeds,episodes=len(seeds),commands=10*len(seeds))))
            def poll(self):
                self.polls+=1
                if self.polls<=3:return None
                self.returncode=0;return 0
            def wait(self,timeout=None):self.returncode=0;return 0
        def clean(cmd,**kw):
            assert cmd==['bash','linux-env/scripts/collect_sim.sh','down','2']
            fd=int(kw['env']['SOARM_INSTANCE_CLAIM_FD']);assert kw['pass_fds']==(fd,)
            for path in reversed(paths):stop(path)
            paths.clear()
            return SimpleNamespace(returncode=0)
        def monotonic():tick[0]+=1.;return tick[0]
        fake_subprocess=SimpleNamespace(Popen=FakeKit,run=clean,check_output=lambda *a,**kw:'3700, 20470',STDOUT=subprocess.STDOUT,TimeoutExpired=subprocess.TimeoutExpired)
        fake_time=SimpleNamespace(time=time.time,time_ns=time.time_ns,strftime=time.strftime,gmtime=time.gmtime,monotonic=monotonic,sleep=lambda _:None)
        deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%M:%SZ')
        argv=[str(folder/'run_native_replay_window.py'),'--out',str(out),'--stop-by-utc',deadline,'--matched-reset-qualification','--bank',str(base/'bank'),'--bank-sha256','fixture','--reset-collector-root',str(repo)]
        if scenario=='qualification-subset-refusal':argv+=['--episode-seeds','530000']
        if scenario=='occupied-instance':
            path=instance/'control.pid'
            start(path,instance/'prior-control.log',[sys.executable,'-c','import time;time.sleep(120)']);paths.append(path)
        if scenario=='claimed-instance':
            from run_smoke import InstanceClaim
            external_claim=InstanceClaim(instance)
        try:
            with mock.patch.object(owner,'REPO',repo),mock.patch.object(owner,'preflight_sources',return_value={'run_episodes.py':'fixture'}),mock.patch.object(owner,'qualified_host',return_value=dict(root=str(host))),mock.patch.object(owner,'subprocess',fake_subprocess),mock.patch.object(owner,'time',fake_time),mock.patch.object(native_bank,'load_bank',return_value=([dict(seed=seed) for seed in seeds],suite)),mock.patch.object(matched_reset,'validate_collector_root'),mock.patch.object(gpu_device,'inventory',return_value=cards),mock.patch.object(nvml_processes,'sample',side_effect=nvml),mock.patch.object(sys,'argv',argv):
                error=None
                try:owner.main()
                except AssertionError:
                    if scenario!='qualification-subset-refusal':raise
                    assert not out.exists() and not calls
                    print(json.dumps(dict(status='PASS',scenario=scenario,refused_before_output_and_launch=True)))
                    return
                except RuntimeError as failure:
                    if scenario in ('pass','late-benign-graphics','seen15'):raise
                    error=str(failure)
            ledger=json.loads((out/'window-ledger.json').read_text())
            assert ledger['cleanup_exit_code']==0 and ledger['stopped_utc']
            assert not paths or scenario=='occupied-instance'
            samplefile=out/'gpu-memory.jsonl'
            samples=[json.loads(l) for l in samplefile.read_text().splitlines()] if samplefile.exists() else []
            if scenario in ('pass','late-benign-graphics','seen15'):
                assert ledger['status']=='matched-reset-qualified' and error is None
                assert any(s.get('owned_mib')==3700 for s in samples)
                assert not any('audit_error' in s for s in samples)
                assert ledger['external_graphics_baseline']['owned'] is False
                assert len(calls)==2
                if scenario=='late-benign-graphics':
                    assert any(any(r['pid']==1 and r['memory_mib']==3 for r in sample.get('benign_foreign_graphics_rows',[])) for sample in samples)
            elif scenario=='prelaunch-compute':
                assert ledger['status']=='stopped-with-error' and 'before launch' in error and not calls
            elif scenario in ('occupied-instance','claimed-instance'):
                assert ledger['status']=='stopped-with-error' and not calls
                assert ledger['cleanup_skipped_no_owned_launch'] is True
                if scenario=='occupied-instance':
                    from managed_process import verified
                    assert verified(paths[0]) is not None,'Pre-existing process was stopped'
                else:assert 'already claimed' in error
            else:
                assert scenario=='late-graphics' and ledger['status']=='stopped-with-error'
                assert 'unverified for3samples' in error and len(calls)==2
                assert sum('audit_error' in s for s in samples)==3
            assert 'torch' not in sys.modules,'Owner CPU preflight must not import model/CUDA'
            print(json.dumps(dict(status='PASS',scenario=scenario,entrypoint=str(folder/'run_native_replay_window.py'),scope='actual owner CLI lifecycle; fake Kit/collector/NVML, no model/GPU',samples=len(samples),baseline_before_launch=scenario in ('pass','late-graphics'),cleanup=0,torch_imported=False)))
        finally:
            for path in reversed(paths):stop(path)
            if external_claim is not None:external_claim.close()

class OwnerEntrypointTests(unittest.TestCase):
    def run_scenario(self,scenario):
        xml=Path(__file__).with_name('cyclonedds-loopback-qualified.xml')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--exercise',str(xml),scenario],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('"scenario": "'+scenario+'"',result.stdout)
    def test_clean_interpreter_seen15_owner_lifecycle(self):self.run_scenario('seen15')
    def test_clean_interpreter_actual_owner_lifecycle(self):self.run_scenario('pass')
    def test_prelaunch_compute_refusal_still_records_stop_and_cleanup(self):self.run_scenario('prelaunch-compute')
    def test_new_graphics_after_launch_refuses_three_times_and_cleans(self):self.run_scenario('late-graphics')
    def test_benign_graphics_birth_after_launch_passes(self):self.run_scenario('late-benign-graphics')
    def test_qualification_rejects_resume_subset_before_launch(self):self.run_scenario('qualification-subset-refusal')
    def test_preexisting_managed_process_survives_refused_window(self):self.run_scenario('occupied-instance')
    def test_preexisting_instance_claim_survives_refused_window(self):self.run_scenario('claimed-instance')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--exercise':exercise(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'pass')
    else:unittest.main()
