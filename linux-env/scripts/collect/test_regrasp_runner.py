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


class Fake:
    def __init__(self,outcomes=(),fail_down=False,fail_export=False):
        self.outcomes=iter(outcomes);self.calls=[];self.fail_down=fail_down;self.fail_export=fail_export
    def verify_prior(self,state):self.calls.append(('verify',))
    def start(self,c):self.calls.append(('start',c['index'],c['base_seed']))
    def attempt(self,c,t,s):
        self.calls.append(('attempt',c['index'],t,s))
        value=next(self.outcomes,'accepted')
        if value=='terminal':raise RuntimeError('Post-writer fault')
        return dict(kind=value)
    def between(self,c,t):self.calls.append(('between',c['index'],t))
    def stop(self):self.calls.append(('down',));return down(not self.fail_down)
    def export(self,c):
        self.calls.append(('export',c['index']))
        if self.fail_export:raise RuntimeError('Corrupt decoded video')
        return exported()


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
    def test_pending_or_stopped_or_different_resume_refuses(self):
        r,a,s=self.run_engine();r.run()
        for key,value in [('pending',{'trial':0}),('stopped',True),('config',{})]:
            state=copy.deepcopy(s[-1]);state[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):r.run(state)


class EvidenceTests(unittest.TestCase):
    def make(self,kind='baseline'):
        def disturb(h,frame,step):
            if frame==0 and step==1:h.bodies[h.neighbour]['velocity'][0]=POLICY['stationary_linear_m_s']*2
        h=SetupHarness(disturb=disturb if kind=='substep' else None)
        if kind=='baseline':h.bodies[h.neighbour]['velocity'][0]=POLICY['stationary_linear_m_s']*2
        with patch.dict('sys.modules',{'collection_runtime':h.proxy}):
            try:h.collector._prepare_regrasp_start(h.params,h.result)
            except RuntimeError as error:h.runtime.fail_miss_motion(error)
            else:raise AssertionError('Expected real protected screen')
        fault=h.journal[-1];report=h.result['recovery_report'];report['fault_epoch']=fault['epoch']
        receipt=dict(params=dict(h.params,trial=0),accepted=False,episode_id='fixture',recovery_report=report)
        return receipt,fault,h
    def test_real_precondition_and_substep_screen_proofs_classify(self):
        for kind in ['baseline','substep']:
            r,f,h=self.make(kind)
            with self.subTest(kind=kind):
                p=screen_evidence(r,f);self.assertEqual(p['reason'],'protected_velocity_screen')
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
            with self.subTest(field=field),self.assertRaises((RuntimeError,ValueError)):screen_evidence(r,bad)
    def test_below_screen_and_unpublished_fault_refuse(self):
        r,f,h=self.make()
        for field,value in [('durability_unresolved',True),('dirty',True),('pause_error','failed'),('publication_attempts',0)]:
            bad=copy.deepcopy(f);bad[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):screen_evidence(r,bad)
        bad=copy.deepcopy(r);s=bad['recovery_report']['baseline'];s['bodies'][h.neighbour]['velocity'][0]=0
        from test_tick_contract import checked_window
        bad['recovery_report']['baseline']=checked_window([s])[0]
        from recovery_start import world_anchor
        bad['recovery_report']['world_anchor_sha256']=digest(world_anchor(bad['recovery_report']['baseline']))
        with self.assertRaisesRegex(RuntimeError,'No measured'):screen_evidence(bad,f)
    def fixture_files(self,root,kind='baseline'):
        r,f,h=self.make(kind);run=root/'cohort/run';path=run/'attempt-0000';path.mkdir(parents=True)
        episode_id=__import__('hashlib').sha256((str(run)+':0:'+str(r['params']['seed'])).encode()).hexdigest()
        r['episode_id']=episode_id;r['recovery_report']['episode_id']=episode_id
        runtime=root/'runtime';native=runtime/'native-motion-faults/test.json';native.parent.mkdir(parents=True)
        native.write_text(json.dumps(f));message=semantic_error(f['error'])
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
            run,path,r,runtime=self.fixture_files(Path(d));stage=dict(pid=1234,started_wall_ns=90)
            self.assertEqual(classify(run,0,r['params']['seed'],stage,runtime)['kind'],'screen_reject')
            for name in ['frames.jsonl','wrist-encode.log','top.mp4']:
                p=path/name;p.touch()
                with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'writer'):classify(run,0,r['params']['seed'],stage,runtime)
                p.unlink()
            with self.assertRaisesRegex(RuntimeError,'retirement'):classify(run,0,r['params']['seed'],dict(pid=1235,started_wall_ns=90),runtime)
            native=Path(r['native_miss_fault_file']);native.unlink();native.symlink_to(path/'receipt.json')
            with self.assertRaises((OSError,RuntimeError)):classify(run,0,r['params']['seed'],stage,runtime)
    def test_cleanup_errors_and_changed_summary_refuse(self):
        for change in ['cleanup','summary','second_cause','late_writer','retirement']:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                run,path,r,runtime=self.fixture_files(Path(d));stage=dict(pid=1234,started_wall_ns=90)
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
            outcome=classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90),runtime)
            self.assertEqual(outcome['kind'],'screen_reject');self.assertIn('protected_settle',outcome['native_fault'])
    def test_changed_episode_identity_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));r['episode_id']='other'
            (path/'receipt.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(RuntimeError,'identity'):classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90),runtime)
    def test_second_terminal_cause_and_changed_fault_refuse(self):
        with tempfile.TemporaryDirectory() as d:
            run,path,r,runtime=self.fixture_files(Path(d));r['capture_failures']=[dict(error='camera gap')]
            (path/'receipt.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(RuntimeError,'Camera'):classify(run,0,r['params']['seed'],dict(pid=1234,started_wall_ns=90),runtime)


if __name__=='__main__':unittest.main()
