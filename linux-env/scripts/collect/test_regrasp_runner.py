"""Outer recovery policy uses real native setup proofs; no Isaac or signals."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from regrasp_runner import Runner,classify,screen_evidence,require_down,digest,POLICY,semantic_error
from test_protected_settle import SetupHarness
from contract import parameters
from storage import CANONICAL_ROOT


def down(passed=True):
    return dict(passed=passed,remaining=[],ports_listening={'8783':False,'8784':False})


def exported():
    return dict(passed=True,loaded=True,fps=30,episodes=3,timing_passed=True,frames=300,
        frame_phases=dict(counts=dict(normal=0,miss_prefix=0,corrective=300)),
        export_content=dict(metadata=dict(passed=True),video=dict(passed=True,scope='ALL frames',decoded_frame_pairs=600)))


class StandaloneTests(unittest.TestCase):
    def clean_process(self,argv,cwd):
        import os,subprocess,sys
        env=dict(os.environ,PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-B',*argv],env=env,cwd=cwd,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return json.loads(result.stdout)
    def test_documented_plan_with_empty_pythonpath(self):
        import subprocess
        from regrasp_runner import REPO
        cut=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='runner-bootstrap-') as d:
            output=Path(d)/'standalone-plan'
            value=self.clean_process(['linux-env/scripts/collect/regrasp_runner.py','--plan',
                '--instance','3','--gpu','0','--cohorts','50','--seed-start','600000','--seed-end','604239',
                '--cut',cut,'--output',str(output)],REPO)
            self.assertEqual(value['cohorts'],50);self.assertEqual(value['seed_start'],600000)
            self.assertEqual(value['cut'],cut);self.assertFalse(output.exists())
    def test_cpu_cohort_dry_path_with_empty_pythonpath(self):
        from regrasp_runner import REPO
        entry=REPO/'linux-env/scripts/collect/regrasp_runner.py'
        # A direct script gets its own directory on sys.path; run_path needs the
        # same script-directory entrance, never the control-package directory.
        code="""import json,runpy,sys
from pathlib import Path
entry=Path(sys.argv[1]);sys.path.insert(0,str(entry.parent))
runner=runpy.run_path(str(entry))
import so_arm101_control.grasp_geometry as geometry
expected=entry.parents[3]/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/grasp_geometry.py'
assert Path(geometry.__file__).resolve()==expected.resolve()
from test_regrasp_runner import Fake
config=dict(cohorts=1,seed_start=600000,seed_end=600002)
saved=[];state=runner['Runner'](config,Fake(),saved.append).run()
assert state['complete'] and not state['stopped']
print(json.dumps(dict(selected=state['selected_episodes'],colors=state['color_counts'],source=str(Path(geometry.__file__).resolve()),cpu_only=True)))
"""
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='runner-bootstrap-') as d:
            value=self.clean_process(['-c',code,str(entry)],Path(d))
        self.assertEqual(value['selected'],3);self.assertEqual(value['colors'],dict(red=1,green=1,blue=1))
        self.assertTrue(value['cpu_only'])

class NativeBindingTests(unittest.TestCase):
    def fake(self,spawn,current):
        import ast,hashlib,re
        from types import SimpleNamespace as NS
        from regrasp_runner import Processes,REPO
        p=object.__new__(Processes)
        p.ledger=NS(scan=lambda:None,known={spawn['pid']:copy.deepcopy(spawn)},handles={spawn['pid']:42})
        record=dict(current,runtime_module=str((REPO/'linux-env/scripts/collect/collection_runtime.py').resolve()),
            loaded_sha256=hashlib.sha256((REPO/'linux-env/scripts/collect/collection_runtime.py').read_bytes()).hexdigest())
        def call(code):
            nonce=ast.literal_eval(re.search("'nonce':([^,}]+)",code).group(1))
            return dict(record,nonce=nonce)
        p.native_call=call
        return p,record
    def check(self,p,current):
        with patch('regrasp_runner.identity',return_value=dict(current,state='S')),\
                patch('regrasp_runner.os.dup',return_value=43),patch('regrasp_runner.os.close'),\
                patch('regrasp_runner.select.select',return_value=([],[],[])):
            return p.observe('c'*32)
    def test_real_receipt_group_transitions_keep_owned_birth(self):
        from regrasp_evidence import require_process_binding
        f=json.loads((Path(__file__).with_name('fixtures')/'native-process-transitions.json').read_text())
        self.assertEqual(len(f['slots']['3']['transitions']),2)
        for change in f['slots']['3']['transitions']:
            p,record=self.fake(change['spawn'],change['later'])
            value=self.check(p,change['later']);require_process_binding(value)
            self.assertEqual(value['owned_identity'],change['later'])
            self.assertEqual(value['process_binding']['spawn_identity'],change['spawn'])
            self.assertNotEqual(value['owned_identity']['pgid'],value['process_binding']['spawn_identity']['pgid'])
    def test_changed_birth_foreign_pid_nonce_or_dead_pidfd_refuses(self):
        current=dict(pid=99,start_ticks='birth',pgid=99)
        for change in ('birth','foreign','nonce','fd_before','fd_after','kernel_after','group_during','closed_fd'):
            with self.subTest(change=change):
                p,record=self.fake(dict(current,pgid=98),current)
                if change=='birth':record['start_ticks']='changed'
                if change=='foreign':record['pid']=100
                if change=='nonce':p.native_call=lambda code:dict(record,nonce='wrong')
                with patch('regrasp_runner.identity',return_value=dict(current,state='S')) as kernel,\
                        patch('regrasp_runner.os.dup',return_value=43) as duplicate,patch('regrasp_runner.os.close'),\
                        patch('regrasp_runner.select.select',return_value=([],[],[])) as ready:
                    if change=='fd_before':ready.return_value=([43],[],[])
                    if change=='fd_after':ready.side_effect=[([],[],[]),([43],[],[])]
                    if change=='kernel_after':kernel.side_effect=[dict(current,state='S'),None]
                    if change=='group_during':kernel.side_effect=[dict(current,state='S'),dict(current,state='S',pgid=100)]
                    if change=='closed_fd':duplicate.side_effect=OSError('released pidfd')
                    with self.assertRaises((RuntimeError,OSError)):p.observe('c'*32)
    def test_refusal_durably_records_spawn_and_current_identities(self):
        current=dict(pid=99,start_ticks='birth',pgid=99)
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='native-binding-') as d:
            p,_=self.fake(dict(current,pgid=98),current);p.output=Path(d)
            with patch('regrasp_runner.identity',side_effect=[dict(current,state='S'),dict(current,state='S',pgid=100)]),\
                    patch('regrasp_runner.os.dup',return_value=43),patch('regrasp_runner.os.close'),\
                    patch('regrasp_runner.select.select',return_value=([],[],[])):
                with self.assertRaises(RuntimeError):p.observe('c'*32)
            files=list((Path(d)/'native-observation-failures').glob('*.json'));self.assertEqual(len(files),1)
            proof=json.loads(files[0].read_text());self.assertEqual(proof['spawn_identity']['pgid'],98)
            self.assertEqual(proof['kernel_before']['pgid'],99);self.assertEqual(proof['kernel_after']['pgid'],100)
            self.assertEqual(proof['response_identity']['nonce'],'c'*32)
            self.assertNotIn('diagnostic_publication_error',proof)
    def test_live_owned_child_setsid_then_nonce_observation(self):
        import subprocess,sys,hashlib,time
        from regrasp_runner import Processes,REPO
        from run_smoke import OwnedProcesses,identity
        from regrasp_evidence import require_process_binding
        module=REPO/'linux-env/scripts/collect/collection_runtime.py'
        code="""import hashlib,json,os,sys,types
from pathlib import Path
module=Path(sys.argv[1]);r=types.SimpleNamespace(epoch=1,stamp=lambda:1000000000,native_miss_fault_path='test-only',failure=None,busy=False,timeline=types.SimpleNamespace(is_playing=lambda:False),snapshot=lambda:{'epoch':1,'sim_ns':1000000000})
c=types.ModuleType('collection_runtime');c.__file__=str(module);c._LOADED_SHA256=hashlib.sha256(module.read_bytes()).hexdigest();c.runtime=lambda:r
sys.modules['collection_runtime']=c
sys.path.insert(0,str(module.parent))
from test_stage_readiness import StageFixture
fixture=StageFixture();r.body_paths=fixture.paths
for name,value in fixture.modules.items():
 if name!='collection_runtime':sys.modules[name]=value
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request.get('setsid'):
  os.setsid();print(json.dumps({'pgid':os.getpgid(0)}),flush=True);continue
 scope={};exec(request['code'],scope);print(json.dumps(scope['result']),flush=True)
"""
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='native-binding-') as d:
            process=subprocess.Popen([sys.executable,'-B','-u','-c',code,str(module)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            p=object.__new__(Processes);p.output=Path(d);p.ledger=OwnedProcesses(Path(d),time.time_ns())
            try:
                self.assertTrue(json.loads(process.stdout.readline())['ready']);p.ledger.add(process.pid)
                old=copy.deepcopy(p.ledger.known[process.pid])
                process.stdin.write(json.dumps(dict(setsid=True))+'\n');process.stdin.flush()
                self.assertEqual(json.loads(process.stdout.readline())['pgid'],process.pid)
                self.assertNotEqual(old['pgid'],identity(process.pid)['pgid'])
                def call(code):
                    process.stdin.write(json.dumps(dict(code=code))+'\n');process.stdin.flush()
                    return json.loads(process.stdout.readline())
                p.native_call=call;value=p.observe('c'*32);require_process_binding(value)
                self.assertEqual(value['pid'],process.pid);self.assertEqual(value['process_binding']['spawn_identity']['pgid'],old['pgid'])
                self.assertEqual(value['owned_identity']['pgid'],process.pid)
                ready=p.observe('e'*32,readiness=True);require_process_binding(ready)
                from stage_readiness import errors
                self.assertEqual(errors(ready['stage_readiness']),[])
                self.assertEqual(ready['process_binding']['request_nonce'],'e'*32)
            finally:
                process.stdin.close();process.wait(timeout=5);process.stdout.close();process.stderr.close()
                for fd in p.ledger.handles.values():__import__('os').close(fd)
    def test_stored_binding_must_keep_pidfd_and_nonce_proof(self):
        from regrasp_evidence import require_process_binding
        current=dict(pid=99,start_ticks='birth',pgid=99)
        p,_=self.fake(dict(current,pgid=98),current);original=self.check(p,current)
        for keys in (('pidfd','alive_after_rpc'),('kernel_before','pgid'),('spawn_identity','start_ticks'),('request_nonce',)):
            bad=copy.deepcopy(original);x=bad['process_binding']
            for key in keys[:-1]:x=x[key]
            x[keys[-1]]=False
            with self.subTest(keys=keys),self.assertRaises(RuntimeError):require_process_binding(bad)


class Fake:
    def __init__(self,outcomes=(),fail_down=False,fail_export=False):
        self.outcomes=iter(outcomes);self.calls=[];self.fail_down=fail_down;self.fail_export=fail_export;self.facts={};self.attempts={}
    def verify_prior(self,state):
        self.calls.append(('verify',))
        return [copy.deepcopy(self.facts[i]) for i in sorted(self.facts)]
    def sealed(self,c,kind):
        value=dict(index=c['index'],kind=kind,events=copy.deepcopy(c['events']),attempts=copy.deepcopy(self.attempts[c['index']]))
        self.facts[c['index']]=value
        return digest(value)
    def quarantine(self,c):return self.sealed(c,'quarantined')
    def start(self,c):self.calls.append(('start',c['index'],c['base_seed']))
    def attempt(self,c,t,s):
        self.calls.append(('attempt',c['index'],t,s))
        value=next(self.outcomes,'accepted')
        if value=='terminal':raise RuntimeError('Post-writer fault')
        record=dict(params=dict(parameters(s,('red','green','blue')[t],regrasp=True),trial=t),accepted=value=='accepted')
        record['receipt_sha256']=digest(record)
        self.attempts.setdefault(c['index'],[]).append(copy.deepcopy(record))
        return dict(kind=value,receipt_sha256=record['receipt_sha256'])
    def between(self,c,t):self.calls.append(('between',c['index'],t))
    def stop(self):self.calls.append(('down',));return down(not self.fail_down)
    def export(self,c):
        self.calls.append(('export',c['index']))
        if self.fail_export:raise RuntimeError('Corrupt decoded video')
        return dict(exported(),runner_artifact_seal_sha256=self.sealed(c,'selected'))


class EngineTests(unittest.TestCase):
    def run_engine(self,outcomes=(),cohorts=1,**kw):
        adapter=Fake(outcomes,**kw);saved=[]
        config=dict(cohorts=cohorts,seed_start=600000,seed_end=601000)
        runner=Runner(config,adapter,lambda s:saved.append(copy.deepcopy(s)))
        return runner,adapter,saved
    def test_complete_rgb_is_selected_only_after_down_and_export(self):
        r,a,s=self.run_engine();state=r.run()
        self.assertEqual(state['selected_episodes'],3);self.assertEqual(state['color_counts'],dict(red=1,green=1,blue=1))
        self.assertEqual([c[2] for c in a.calls if c[0]=='attempt'],[0,1,2])
        self.assertLess(a.calls.index(('down',)),a.calls.index(('export',0)))
        self.assertEqual(state['unselected_prefix_episodes'],0);self.assertTrue(state['complete'])
    def test_screen_quarantines_entire_prefix_and_next_seed_starts_fresh(self):
        r,a,s=self.run_engine(['accepted','screen_reject','accepted','accepted','accepted'])
        state=r.run()
        self.assertEqual(state['attempts'],5);self.assertEqual(state['screen_rejects'],1)
        self.assertEqual(state['selected_episodes'],3);self.assertEqual(state['unselected_prefix_episodes'],1)
        self.assertEqual([c for c in a.calls if c[0]=='start'],[('start',0,600000),('start',1,600002)])
        self.assertTrue(state['cohorts'][0]['quarantined']);self.assertFalse(state['cohorts'][0]['selected'])
        self.assertNotIn(('export',0),a.calls);self.assertIn(('export',1),a.calls)
    def test_fault_after_writer_never_retries(self):
        r,a,s=self.run_engine(['accepted','terminal'])
        with self.assertRaisesRegex(RuntimeError,'Post-writer'):r.run()
        self.assertEqual(len([c for c in a.calls if c[0]=='attempt']),2)
        self.assertTrue(s[-1]['stopped']);self.assertEqual(s[-1]['attempts'],2)
        self.assertEqual(s[-1]['unselected_prefix_episodes'],1)
    def test_cleanup_failure_never_retries(self):
        r,a,s=self.run_engine(['screen_reject'],fail_down=True)
        with self.assertRaisesRegex(RuntimeError,'Owned-down'):r.run()
        self.assertEqual(len([c for c in a.calls if c[0]=='start']),1);self.assertFalse(s[-1]['complete'])
    def test_screen_fraction_not_applied_before20_attempts(self):
        r,a,s=self.run_engine(['screen_reject']*19,cohorts=1)
        with self.assertRaisesRegex(RuntimeError,'10percent'):r.run()
        self.assertEqual(s[-1]['attempts'],20);self.assertEqual(len([c for c in a.calls if c[0]=='attempt']),20)
    def test_above10percent_stops_at20_even_after_accepted_attempt(self):
        r,a,s=self.run_engine(['screen_reject']*3+['accepted']*17,cohorts=10)
        with self.assertRaisesRegex(RuntimeError,'10percent'):r.run()
        self.assertEqual(s[-1]['attempts'],20);self.assertTrue(s[-1]['stopped'])
    def test_exactly10percent_continues(self):
        r,a,s=self.run_engine(['screen_reject']*2+['accepted']*18,cohorts=6)
        state=r.run();self.assertEqual(state['attempts'],20);self.assertEqual(state['selected_episodes'],18)
    def test_export_failure_keeps_cohort_unselected(self):
        r,a,s=self.run_engine(fail_export=True)
        with self.assertRaisesRegex(RuntimeError,'Corrupt'):r.run()
        self.assertEqual(s[-1]['selected_episodes'],0);self.assertEqual(s[-1]['unselected_prefix_episodes'],3)
    def test_resume_counters_and_seed_gaps_refuse(self):
        r,a,s=self.run_engine();r.run()
        for key,value in [('attempts',4),('next_seed',600004),('selected_episodes',0),('screen_rejects',1)]:
            state=copy.deepcopy(s[-1]);state[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):r.run(state)
    def test_bad_export_certificate_stops_before_selection(self):
        r,a,s=self.run_engine()
        a.export=lambda c:dict(exported(),loaded=False)
        with self.assertRaisesRegex(RuntimeError,'export'):r.run()
        self.assertEqual(s[-1]['selected_episodes'],0)
    def test_selected_checkpoint_resume_reverifies_prior_artifacts(self):
        r,a,s=self.run_engine();r.run();state=r.run(s[-1])
        self.assertTrue(state['complete']);self.assertIn(('verify',),a.calls)
    def test_all_unknown_outcomes_are_terminal(self):
        r,a,s=self.run_engine(['unknown'])
        with self.assertRaisesRegex(RuntimeError,'Unknown'):r.run()
        self.assertTrue(s[-1]['stopped'])
    def test_cleanup_failure_is_latched_across_subsequent_stop(self):
        from regrasp_runner import Processes
        p=object.__new__(Processes);p.failed_cleanup={'remaining':[{'pid':1}]};p.ledger=None
        with self.assertRaisesRegex(RuntimeError,'Previous owned-down'):p.stop()
    def test_duplicate_orphan_conflicting_or_relabelled_cohorts_refuse(self):
        r,a,s=self.run_engine(cohorts=2);r.run();original=s[-1]
        mutations=[]
        bad=copy.deepcopy(original);bad['cohorts'][1]=copy.deepcopy(bad['cohorts'][0]);mutations.append(bad)
        bad=copy.deepcopy(original);bad['cohorts'][1]['quarantined']=True;mutations.append(bad)
        bad=copy.deepcopy(original);bad['cohorts'][1]['base_seed']=900000;mutations.append(bad)
        bad=copy.deepcopy(original);bad['events']=[];bad['cohorts']=[];bad['attempts']=0;bad['next_seed']=600000;mutations.append(bad)
        bad=copy.deepcopy(original);bad['events'].append(copy.deepcopy(bad['events'][-1]));mutations.append(bad)
        for number,bad in enumerate(mutations):
            with self.subTest(number=number),self.assertRaises(RuntimeError):r.run(bad)
    def test_empty_resume_refuses_before_publishing(self):
        r,a,s=self.run_engine()
        with self.assertRaises((KeyError,RuntimeError)):r.run({})
        self.assertFalse(s)
    def test_foreign_or_reused_launcher_marker_never_runs_down(self):
        from regrasp_runner import Processes
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        for saved in ({},{88:dict(start_ticks='old')}):
            with self.subTest(saved=saved),tempfile.TemporaryDirectory() as d:
                p=object.__new__(Processes);p.runtime=Path(d);p.output=Path(d);p.config={'instance':3}
                p.live=True;p.stage_index=1;p.current_cohort=None;p.stage=Mock()
                p.ledger=NS(known=saved,stop=lambda:dict(tracked=[],remaining=[]))
                p.probe_ports=lambda:{'8783':False,'8784':False}
                (p.runtime/'control.pid').write_text('88')
                with patch('regrasp_runner.identity',return_value=dict(pid=88,start_ticks='new')),patch('regrasp_runner.atomic_json'):
                    with self.assertRaises(RuntimeError):p.stop()
                p.stage.assert_not_called();self.assertTrue(p.failed_cleanup)
    def test_no_owner_does_not_assume_foreign_port_closed(self):
        from regrasp_runner import Processes
        p=object.__new__(Processes);p.ledger=None;p.probe_ports=lambda:{'8783':True,'8784':False}
        with self.assertRaisesRegex(RuntimeError,'Owned-down'):p.stop()
    def test_production_on_dual_is_refused_even_with_space(self):
        from regrasp_runner import Processes
        p=object.__new__(Processes)
        with patch('host_config.profile',return_value={'name':'dual-a4500'}):
            with self.assertRaisesRegex(RuntimeError,'only on rtx6000'):p.capacity()
    def test_pending_or_stopped_or_different_resume_refuses(self):
        r,a,s=self.run_engine();r.run()
        for key,value in [('pending',{'trial':0}),('stopped',True),('config',{})]:
            state=copy.deepcopy(s[-1]);state[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):r.run(state)


def fixture_binding(value):
    expected={k:value[k] for k in ('pid','start_ticks','pgid')}
    value['owned_identity']=copy.deepcopy(expected)
    value['process_binding']=dict(version=1,spawn_identity=copy.deepcopy(expected),
        discovery=dict(nonce='b'*32,identity=copy.deepcopy(expected)),kernel_before=copy.deepcopy(expected),
        kernel_after=copy.deepcopy(expected),pidfd=dict(held_before_rpc=True,alive_before_rpc=True,alive_after_rpc=True),
        request_nonce=value['nonce'],response_nonce=value['nonce'])
    return value


class EvidenceTests(unittest.TestCase):
    def boot(self,h):
        return dict(finger_model=copy.deepcopy(h.collector.native_finger_model),effective_shape_offsets=copy.deepcopy(h.collector.native_shape_offsets))
    def make(self,kind='baseline'):
        def disturb(h,frame,step):
            if frame==(1 if kind=='later' else 0) and step==1:h.bodies[h.neighbour]['velocity'][0]=POLICY['stationary_linear_m_s']*2
        h=SetupHarness(disturb=disturb if kind in ('substep','later') else None)
        if kind=='baseline':h.bodies[h.neighbour]['velocity'][0]=POLICY['stationary_linear_m_s']*2
        if kind=='final':
            finish=h.runtime.finish_protected_frame
            def final_screen():
                h.bodies[h.neighbour]['velocity'][0]=POLICY['stationary_linear_m_s']*2
                return finish()
            h.runtime.finish_protected_frame=final_screen
        with patch.dict('sys.modules',{'collection_runtime':h.proxy}):
            try:h.collector._prepare_regrasp_start(h.params,h.result)
            except RuntimeError as error:h.runtime.fail_miss_motion(error)
            else:raise AssertionError('Expected real protected screen')
        fault=h.journal[-1];report=h.result['recovery_report'];report['fault_epoch']=fault['epoch']
        receipt=dict(params=dict(h.params,trial=0),accepted=False,episode_id='fixture',recovery_report=report)
        return receipt,fault,h
    def test_real_precondition_and_substep_screen_proofs_classify(self):
        for kind in ['baseline','substep','final','later']:
            r,f,h=self.make(kind)
            with self.subTest(kind=kind):
                p=screen_evidence(r,f,self.boot(h));self.assertEqual(p['reason'],'protected_velocity_screen')
                self.assertIn(h.neighbour,p['bodies'])
    def test_displacement_clock_and_drive_changes_are_terminal(self):
        r,f,h=self.make('substep')
        for field in ['position','clock','drive','base','nan']:
            bad=copy.deepcopy(f);s=bad['protected_settle']['pending'][-1]['snapshot']
            if field=='position':s['bodies'][h.neighbour]['position'][0]+=.001
            if field=='clock':s['sim_ns']+=10**8
            if field=='drive':s['drive_targets'][0]+=.1
            if field=='base':s['robot_base_world_pose']['position'][0]+=.1
            if field=='nan':s['bodies'][h.neighbour]['position'][0]=float('nan')
            with self.subTest(field=field),self.assertRaises((RuntimeError,ValueError)):screen_evidence(r,bad,self.boot(h))
    def test_below_screen_and_unpublished_fault_refuse(self):
        r,f,h=self.make()
        for field,value in [('durability_unresolved',True),('dirty',True),('pause_error','failed'),('publication_attempts',0)]:
            bad=copy.deepcopy(f);bad[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):screen_evidence(r,bad,self.boot(h))
        bad=copy.deepcopy(r);s=bad['recovery_report']['baseline'];s['bodies'][h.neighbour]['velocity'][0]=0
        from test_tick_contract import checked_window
        bad['recovery_report']['baseline']=checked_window([s])[0]
        bad['recovery_report']['pre_restore_observed']=copy.deepcopy(bad['recovery_report']['baseline'])
        from recovery_start import world_anchor
        bad['recovery_report']['world_anchor_sha256']=digest(world_anchor(bad['recovery_report']['baseline']))
        with self.assertRaisesRegex(RuntimeError,'No measured'):screen_evidence(bad,f,self.boot(h))
    def fixture_files(self,root,kind='baseline'):
        r,f,h=self.make(kind);run=root/'cohort/run';path=run/'attempt-0000';path.mkdir(parents=True)
        episode_id=__import__('hashlib').sha256((str(run)+':0:'+str(r['params']['seed'])).encode()).hexdigest()
        r['episode_id']=episode_id;r['recovery_report']['episode_id']=episode_id
        runtime=root/'runtime';native=runtime/'native-motion-faults/test.json';native.parent.mkdir(parents=True)
        native.write_text(json.dumps(f));message=semantic_error(f['error'])
        stage=dict(pid=1234,started_wall_ns=90,invocation='c'*32)
        native_id=dict(pid=99,start_ticks='native-birth',pgid=99,fault_path=str(native),
            loaded_sha256='source-fingerprint',runtime_module='owned-native-module')
        native_id['owned_identity']={k:native_id[k] for k in ('pid','start_ticks','pgid')}
        boot=self.boot(h);boot.update(native_miss_fault_path=str(native),loaded_sha256=native_id['loaded_sha256'])
        record=dict(transport=boot,native=fixture_binding(dict(native_id,nonce='d'*32)))
        before=dict(native_id,nonce='c'*32,wall_ns=80,epoch=r['recovery_report']['baseline']['epoch']-1,
            sim_ns=r['recovery_report']['baseline']['sim_ns']-33333333,fault=None,failure=None,busy=False,playing=False)
        before['snapshot']=copy.deepcopy(r['recovery_report']['baseline'])
        before['snapshot'].update(epoch=before['epoch'],sim_ns=before['sim_ns'])
        after=dict(native_id,nonce='c'*32,wall_ns=300,epoch=f['epoch']+1,sim_ns=f['sim_ns'],
            fault=copy.deepcopy(f),fault_durable=True,failure=message,busy=False,playing=False,snapshot=h.runtime.snapshot())
        after['snapshot'].update(epoch=after['epoch'],sim_ns=after['sim_ns'])
        fixture_binding(before);fixture_binding(after)
        (run.parent/'native-boot.json').write_text(json.dumps(record))
        (run.parent/'observations').mkdir()
        (run.parent/'observations/attempt-0000.json').write_text(json.dumps(dict(version=1,invocation='c'*32,stage=stage,before=before,after=after)))
        events=[dict(source=s,error=repr(RuntimeError(message)),sensor_evidence=None,wall_ns=150) for s in ['synthetic start verification','attempt disposition','before cleanup','after cleanup']]
        marker=dict(version=1,events=events)
        r.update(error=repr(RuntimeError(message)),camera_boundary=dict(passed=True),camera_publishers={},capture_failures=[],
            native_miss_fault_file=str(native),native_miss_faults=[f],terminal_failure=marker,retirement=dict(passed=False,error=repr(RuntimeError(message))))
        (path/'receipt.json').write_text(json.dumps(r))
        for p in [run/'terminal-fault.json',path/'terminal-fault.json']:p.write_text(json.dumps(marker))
        summary=dict(producer_invocation='a'*32,terminal_failure=marker);(run/'summary.json').write_text(json.dumps(summary))
        certificate=dict(version=1,invocation='a'*32,producer_pid=1234,started_wall_ns=100,completed_wall_ns=200,completed=True,passed=False,cleanup_errors=[],terminal_state='known terminal marker',summary_sha256=__import__('hashlib').sha256((run/'summary.json').read_bytes()).hexdigest())
        (run/'collector-finalization.json').write_text(json.dumps(certificate))
        return run,path,r,runtime
    def test_actual_classification_requires_durable_matching_retirement_and_no_writer(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));stage=dict(pid=1234,started_wall_ns=90,invocation='c'*32)
            self.assertEqual(classify(run,0,r['params']['seed'],stage,runtime)['kind'],'screen_reject')
            for name in ['frames.jsonl','wrist-encode.log','top.mp4']:
                p=path/name;p.touch()
                with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'writer'):classify(run,0,r['params']['seed'],stage,runtime)
                p.unlink()
            with self.assertRaisesRegex(RuntimeError,'retirement|producing stage'):classify(run,0,r['params']['seed'],dict(pid=1235,started_wall_ns=90,invocation='c'*32),runtime)
            native=Path(r['native_miss_fault_file']);native.unlink();native.symlink_to(path/'receipt.json')
            with self.assertRaises((OSError,RuntimeError)):classify(run,0,r['params']['seed'],stage,runtime)
    def test_cleanup_errors_and_changed_summary_refuse(self):
        for change in ['cleanup','summary','second_cause','late_writer','retirement']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                run,path,r,runtime=self.fixture_files(Path(d));stage=dict(pid=1234,started_wall_ns=90,invocation='c'*32)
                if change=='cleanup':
                    p=run/'collector-finalization.json';v=json.loads(p.read_text());v['cleanup_errors']=[{'stage':'executor'}];p.write_text(json.dumps(v))
                if change=='summary':(run/'summary.json').write_text('{}')
                if change=='second_cause':
                    for p in [run/'terminal-fault.json',path/'terminal-fault.json']:
                        v=json.loads(p.read_text());v['events'].append(dict(source='ROS callback',error='other',sensor_evidence=None));p.write_text(json.dumps(v))
                if change=='retirement':
                    r['retirement']={'passed':False,'error':'other worker stalled'};(path/'receipt.json').write_text(json.dumps(r))
                if change=='late_writer':
                    r['timing']={'frames':0};(path/'receipt.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):classify(run,0,r['params']['seed'],stage,runtime)
    def test_full_substep_fault_classifies_only_before_writer(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d),'substep')
            outcome=classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)
            self.assertEqual(outcome['kind'],'screen_reject');self.assertIn('protected_settle',outcome['native_fault'])
    def test_all_symlinks_and_special_entries_refuse_before_classification(self):
        import os
        for name,kind in [('top.mp4','dangling'),('unknown','symlink'),('unknown','fifo')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                run,path,r,runtime=self.fixture_files(Path(d));entry=path/name
                if kind=='dangling':entry.symlink_to(path/'missing')
                elif kind=='symlink':entry.symlink_to(path/'receipt.json')
                else:os.mkfifo(entry)
                with self.assertRaisesRegex(RuntimeError,'Non-regular'):
                    classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)
    def test_parent_directory_symlink_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));alias=Path(d)/'alias';alias.symlink_to(run,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'ancestor|Non-regular'):classify(alias,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)
    def test_changed_episode_identity_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));r['episode_id']='other'
            (path/'receipt.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(RuntimeError,'identity'):classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)
    def test_epoch_and_copied_journal_refuse_at_actual_classifier(self):
        for change in ('epoch','rename','fresh_copy','nonce','birth','source','startup_fault','late_scene','late_cause'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                run,path,r,runtime=self.fixture_files(Path(d));stage=dict(pid=1234,started_wall_ns=90,invocation='c'*32)
                native=Path(r['native_miss_fault_file']);fault=json.loads(native.read_text())
                observation_path=run.parent/'observations/attempt-0000.json';obs=json.loads(observation_path.read_text())
                if change=='epoch':
                    fault['epoch']+=999;r['recovery_report']['fault_epoch']=fault['epoch']
                    r['native_miss_faults']=[fault];native.write_text(json.dumps(fault));obs['after']['fault']=copy.deepcopy(fault)
                if change in ('rename','fresh_copy'):
                    replacement=native.with_name('another-current-file.json');replacement.write_bytes(native.read_bytes())
                    r['native_miss_fault_file']=str(replacement)
                    if change=='fresh_copy':
                        import os
                        stage['started_wall_ns']=1000;obs['stage']=copy.deepcopy(stage)
                        cert_path=run/'collector-finalization.json';cert=json.loads(cert_path.read_text())
                        cert.update(started_wall_ns=1100,completed_wall_ns=1200);cert_path.write_text(json.dumps(cert))
                        for marker_path in (run/'terminal-fault.json',path/'terminal-fault.json'):
                            marker=json.loads(marker_path.read_text())
                            for e in marker['events']:e['wall_ns']=1150
                            marker_path.write_text(json.dumps(marker));r['terminal_failure']=marker
                        os.utime(replacement,None)
                if change=='nonce':obs['after']['nonce']='d'*32
                if change=='birth':obs['after']['start_ticks']='different-birth'
                if change=='source':obs['after']['loaded_sha256']='changed-native-source'
                if change=='startup_fault':obs['before']['fault']=fault
                if change=='late_cause':obs['after']['failure']='another transport failure'
                if change=='late_scene':obs['after']['snapshot']['bodies']['cup_red']['position'][0]+=.1
                observation_path.write_text(json.dumps(obs));(path/'receipt.json').write_text(json.dumps(r))
                with self.assertRaises(RuntimeError):classify(run,0,r['params']['seed'],stage,runtime)
    def test_every_completed_setup_invariant_refuses_at_actual_classifier(self):
        paths=[('selection','proof','passed'),('selection','proof','joints',0),
            ('restore','command_baseline','drive_targets',0),('restore','command_baseline','epoch'),
            ('restore','snapshot','physics_time_s'),('restore','snapshot','velocity',0),
            ('fresh_offsets','epoch'),('fresh_offsets','block_offsets','contact_m',0),
            ('model','all_robot_boxes',0,'half_extents_m',0),('protected_start','sim_ns'),
            ('request','pose','x'),('request','bbox','sx'),
            ('settle_checks',0,'scene_valid'),('settle_checks',0,'static','passed'),
            ('settle_snapshots',0,'tick_contract','row','physics_drive_targets',0)]
        for keys in paths:
            with self.subTest(keys=keys),tempfile.TemporaryDirectory() as d:
                run,path,r,runtime=self.fixture_files(Path(d),'later');value=r['recovery_report']
                for key in keys[:-1]:value=value[key]
                k=keys[-1];value[k]=False if isinstance(value[k],bool) else value[k]+1
                (path/'receipt.json').write_text(json.dumps(r))
                with self.assertRaises((RuntimeError,ValueError)):
                    classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)
    def test_native_observer_verifies_actual_owned_birth_and_source(self):
        current=dict(pid=99,start_ticks='birth',pgid=99)
        for field in ('start_ticks','pgid','runtime_module','loaded_sha256'):
            with self.subTest(field=field):
                p,record=NativeBindingTests().fake(dict(current,pgid=98),current)
                record[field]='wrong'
                with self.assertRaises(RuntimeError):NativeBindingTests().check(p,current)
    def test_second_terminal_cause_and_changed_fault_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));r['capture_failures']=[dict(error='camera gap')]
            (path/'receipt.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(RuntimeError,'Camera'):classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90,invocation='c'*32),runtime)


class ArtifactLedgerTests(unittest.TestCase):
    def on_disk(self,root):
        from regrasp_runner import Processes
        from storage import atomic_json
        import hashlib
        r,a,s=EngineTests().run_engine(cohorts=2);state=r.run()
        p=object.__new__(Processes);p.output=root
        for c in state['cohorts']:
            folder=p.root(c);run=folder/'run';report=folder/'report';report.mkdir(parents=True)
            for trial,record in enumerate(a.attempts[c['index']]):
                receipt=dict(params=record['params'],accepted=record['accepted'])
                path=run/('attempt-%04d'%trial);path.mkdir(parents=True)
                atomic_json(path/'receipt.json',receipt)
                event=c['events'][trial];event['outcome']['receipt_sha256']=digest(receipt)
                state['events'][c['index']*3+trial]=copy.deepcopy(event)
            summary=dict(producer_invocation=('a' if c['index']==0 else 'b')*32)
            atomic_json(run/'summary.json',summary)
            atomic_json(run/'collector-finalization.json',dict(version=1,invocation=summary['producer_invocation'],
                producer_pid=1,started_wall_ns=1,completed_wall_ns=2,completed=True,passed=True,cleanup_errors=[],
                summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()))
            atomic_json(report/'validation.json',exported())
            c['artifact_seal_sha256']=p.seal(c,'selected')
        return r,state,p
    def test_actual_disk_verifier_and_resume_reject_membership_edits(self):
        from regrasp_runner import reconstruct_ledger
        for change in ('none','duplicate','flags','base_seed','empty','orphan','receipt','missing_attempt','sealed_event','bad_seal'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                r,state,p=self.on_disk(Path(d))
                if change=='duplicate':state['cohorts'][1]=copy.deepcopy(state['cohorts'][0])
                if change=='flags':state['cohorts'][1]['quarantined']=True
                if change=='base_seed':state['cohorts'][1]['base_seed']=900000
                if change=='empty':state.update(events=[],cohorts=[],attempts=0,next_seed=600000)
                if change=='orphan':(Path(d)/'cohort-0002').mkdir()
                if change=='receipt':(p.root(state['cohorts'][1])/'run/attempt-0002/receipt.json').write_text('{}')
                if change=='missing_attempt':(p.root(state['cohorts'][1])/'run/attempt-0003').mkdir()
                if change=='sealed_event':state['events'][0]['seed']+=20
                if change=='bad_seal':state['cohorts'][0]['artifact_seal_sha256']='wrong'
                p.stop=lambda:down();r.adapter=p
                if change=='none':
                    facts=p.verify_prior(state);manifest=reconstruct_ledger(state,facts)
                    self.assertEqual([x['seed_start'] for x in manifest],[600000,600003])
                    self.assertTrue(r.run(state)['complete'])
                else:
                    with self.assertRaises(RuntimeError):r.run(state)


if __name__=='__main__':unittest.main()
