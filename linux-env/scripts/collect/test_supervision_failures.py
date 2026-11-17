"""Public certification and original-cause propagation with real CPU workloads."""
import ctypes,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import managed_process as mp
from run_smoke import identity


@unittest.skipUnless(sys.platform=='linux' and hasattr(os,'pidfd_open'),'Linux ownership receipts required')
class SupervisionFailureTests(unittest.TestCase):
    def environment(self):return dict(os.environ,PYTHONPATH=str(Path(mp.__file__).parent))
    def cleanup_workload(self,row):
        old=row['workload'];fd=os.pidfd_open(old['pid'])
        try:
            now=identity(old['pid'])
            if now and now['start_ticks']==old['start_ticks']:signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:os.close(fd)
        os.waitpid(old['pid'],0)
    def abrupt(self,root):
        self.assertEqual(ctypes.CDLL(None).prctl(36,1,0,0,0),0)
        code="""import os,sys
from pathlib import Path
import managed_process as mp
mp.OwnedProcesses.scan=lambda self:os._exit(99)
mp.supervise(Path(sys.argv[1])/'job.pid','abrupt-test',[sys.executable,'-c','import time;time.sleep(90)'])
"""
        p=subprocess.Popen([sys.executable,'-c',code,str(root)],env=self.environment(),start_new_session=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        p.wait(timeout=8);self.assertEqual(p.returncode,99)
        return json.loads(mp.record_path(root/'job.pid').read_text())

    def test_recorded_live_workload_refuses_stop_status_start_with_or_without_pid_marker(self):
        for marker in (True,False):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as d:
                root=Path(d);path=root/'job.pid';row=self.abrupt(root)
                try:
                    self.assertNotIn('children',row)
                    if not marker:path.unlink()
                    self.assertEqual(identity(row['workload']['pid'])['start_ticks'],row['workload']['start_ticks'])
                    result=mp.status(path)
                    self.assertFalse(result['stopped']);self.assertFalse(result['process_exited'])
                    self.assertFalse(result['supervision_ok']);self.assertIn(row['workload']['pid'],[x['pid'] for x in result['remaining']])
                    with self.assertRaises(mp.SupervisionFailure):mp.stop(path)
                    with patch.object(mp.subprocess,'Popen') as spawn:
                        with self.assertRaises(mp.SupervisionFailure):mp.start(path,root/'new.log',['unused'])
                        spawn.assert_not_called()
                    for operation in ('status','stop'):
                        command=subprocess.run([sys.executable,mp.__file__,operation,'--pidfile',str(path)],env=self.environment(),capture_output=True,text=True)
                        self.assertEqual(command.returncode,2,command.stderr)
                        self.assertFalse(json.loads(command.stdout)['process_exited'])
                finally:self.cleanup_workload(row)
                # Gone descendants do not convert an unfinished running receipt
                # into a successful supervision certificate.
                result=mp.status(path);self.assertTrue(result['process_exited']);self.assertFalse(result['supervision_ok'])
                with self.assertRaisesRegex(mp.SupervisionFailure,'incomplete'):mp.stop(path)

    def test_default_sigint_production_workload_keeps_supervisor_cause_after_supervisor_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'job.pid'
            # The workload enters the ACTUAL production coordinate() and its
            # exception/finalization path. Only host/disk/GPU/clock admission and
            # the first wave's body are fake; no ROS, Kit or GPU is launched.
            worker="""import json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import triplet_production as prod
root=Path(sys.argv[1]);release=root/'release.json';release.write_text('{"passed":true}')
real_time=time;clock=[0]
def monotonic():clock[0]+=120;return clock[0]
def wave(_):
 root.joinpath('ready').write_text('ready')
 while True:real_time.sleep(.02)
recipe=dict(cut='a'*40,production_root=str(root/'production'),runtime_root=str(root/'runtime'))
with patch.object(prod,'require_host'),patch.object(prod,'frozen'),patch.object(prod,'disk',return_value={}),patch.object(prod,'confined',side_effect=Path),patch.object(prod,'gpu_uuid',return_value=prod.ASSIGNED_UUID),patch.object(prod.subprocess,'check_output',return_value=''),patch.object(prod,'reduce_waves',side_effect=wave),patch.object(prod,'time',NS(monotonic=monotonic,time_ns=real_time.time_ns,sleep=real_time.sleep)):
 try:prod.coordinate(recipe,release)
 except BaseException:pass
"""
            code="""import sys,json
from pathlib import Path
import managed_process as mp
root=Path(sys.argv[1]);worker=sys.argv[2];original=mp.OwnedProcesses.scan;tripped=[]
def scan(self):
 original(self)
 if root.joinpath('ready').exists() and not tripped:
  tripped.append(True);raise RuntimeError('real supervision scan fault')
mp.OwnedProcesses.scan=scan
mp.supervise(root/'job.pid','real-failure',[sys.executable,'-c',worker,str(root)])
"""
            with (root/'log').open('w') as log:
                process=subprocess.Popen([sys.executable,'-c',code,str(root),worker],env=self.environment(),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
                process.wait(timeout=20)
            self.assertNotEqual(process.returncode,0)
            row=json.loads(mp.record_path(path).read_text());self.assertEqual(row['state'],'failed')
            self.assertIn('real supervision scan fault',row['supervision_error']);self.assertEqual(row['cleanup']['remaining'],[])
            status=mp.status(path);self.assertTrue(status['process_exited']);self.assertFalse(status['supervision_ok'])
            with self.assertRaises(mp.SupervisionFailure) as error:mp.stop(path)
            self.assertEqual(error.exception.report['supervision_error'],row['supervision_error'])
            self.assertTrue(error.exception.report['process_exited']);self.assertFalse(error.exception.report['stopped'])
            for operation in ('status','stop'):
                result=subprocess.run([sys.executable,mp.__file__,operation,'--pidfile',str(path)],env=self.environment(),capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('real supervision scan fault',json.loads(result.stdout)['supervision_error'])
            for filename in ('STOP.json','production-finalization.json'):
                report=json.loads((root/'production'/filename).read_text())
                self.assertIn('real supervision scan fault',report['error'])
                failure=report if filename=='STOP.json' else report['supervision_failure']
                self.assertEqual(failure['execution_error'],'KeyboardInterrupt()')
                self.assertEqual(failure['supervision']['nonce'],'real-failure')
                self.assertEqual(failure['supervision']['supervisor']['pid'],process.pid)
                self.assertTrue(failure['supervision']['signal_intents'])
            final=json.loads((root/'production/production-finalization.json').read_text())
            self.assertFalse(final['passed']);self.assertEqual(final['cleanup']['remaining'],[])

    def fixture(self,root):
        path=root/'job.pid';sup=identity(os.getpid());work=identity(os.getpid())
        row=dict(supervisor=sup,workload=work,nonce='current',pidfile=str(path),state='failed',
            supervision_error='ORIGINAL',cleanup_error='CLEANUP',cleanup=dict(tracked=[],remaining=[]),signal_journal=str(path)+'.signals.jsonl')
        mp.save(mp.record_path(path),row)
        context=dict(receipt_path=str(mp.record_path(path)),nonce='current',supervisor=sup,workload=work,observer=work)
        return path,row,context

    def test_failure_report_binds_receipt_nonce_birth_and_journal_before_retaining_both_errors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,row,context=self.fixture(root)
            event=dict(nonce='current',sender=row['supervisor'],target=row['workload'],signal=2,pidfile=str(path))
            Path(row['signal_journal']).write_text(json.dumps(event)+'\n')
            result=mp.failure_report(context,KeyboardInterrupt())
            self.assertIn('ORIGINAL',result['error']);self.assertEqual(result['execution_error'],'KeyboardInterrupt()')
            self.assertEqual(result['supervision']['cleanup_error'],'CLEANUP')
            self.assertEqual(result['supervision']['signal_intents'],[event])
            for field in ('nonce','supervisor','workload'):
                changed=json.loads(json.dumps(row))
                if field=='nonce':changed[field]='other'
                else:changed[field]['start_ticks']='other'
                mp.save(mp.record_path(path),changed)
                report=mp.failure_report(context,KeyboardInterrupt())
                self.assertIn('evidence unavailable',report['error']);self.assertEqual(report['execution_error'],'KeyboardInterrupt()')
            mp.save(mp.record_path(path),row)
            Path(row['signal_journal']).write_text('{')
            result=mp.failure_report(context,KeyboardInterrupt())
            self.assertIn('ORIGINAL',result['error']);self.assertTrue(result['supervision']['signal_evidence_error'])

    def test_production_worker_stop_and_finalization_keep_bound_original_and_cleanup_causes(self):
        import triplet_production as prod
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,row,context=self.fixture(root)
            worker=object.__new__(prod.Worker);worker.root=root;worker.global_root=root;worker.supervision=context
            worker.plan={};worker.instance=3;worker.wave=0;worker.cut='a'*40;worker.session=0;worker.claim=object();worker.active_job=None
            worker.check=lambda:(_ for _ in ()).throw(KeyboardInterrupt())
            worker.down=lambda:None
            with patch.object(prod,'backup_terminal',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'ORIGINAL'):worker.run([{}])
            for name in ('STOP.json','worker-finalization.json'):
                report=json.loads((root/name).read_text());self.assertIn('ORIGINAL',report['error'])
                failure=report if name=='STOP.json' else report['supervision_failure']
                self.assertEqual(failure['supervision']['cleanup_error'],'CLEANUP')

    def test_dual_failure_public_stop_status_preserve_unknown_cleanup_and_errors_with_missing_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,row,context=self.fixture(root)
            row['supervisor']=dict(row['supervisor'],pid=99999999)
            row['workload']=dict(row['workload'],pid=99999998)
            row['state']='cleanup_failed';row['cleanup']['remaining']=[dict(unverified='cannot enumerate')]
            mp.save(mp.record_path(path),row)
            result=mp.status(path)
            self.assertIsNone(result['process_exited']);self.assertFalse(result['stopped'])
            self.assertEqual(result['supervision_error'],'ORIGINAL');self.assertEqual(result['cleanup_error'],'CLEANUP')
            with self.assertRaises(mp.SupervisionFailure) as error:mp.stop(path)
            self.assertEqual(error.exception.report['cleanup_error'],'CLEANUP')
