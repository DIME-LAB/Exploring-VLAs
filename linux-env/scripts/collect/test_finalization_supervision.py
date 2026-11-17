"""Retirement-time supervisor evidence at the actual coordinator/worker exits."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import managed_process as mp
import triplet_production as prod
from run_smoke import identity


class FinalizationSupervisionTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'job.pid';born=identity(os.getpid())
        row=dict(supervisor=born,workload=born,nonce='owned',pidfile=str(path),state='running',
            supervision_error=None,cleanup_error=None,signal_journal=str(path)+'.signals.jsonl')
        mp.save(mp.record_path(path),row)
        return path,row,dict(receipt_path=str(mp.record_path(path)),nonce='owned',supervisor=born,workload=born)

    def test_refresh_reads_latest_and_keeps_workload_original_and_later_cleanup_causes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,row,context=self.fixture(root)
            previous=mp.failure_report(context,ValueError('initial workload failure'))
            row.update(state='stopping',supervision_error='LATE ORIGINAL',cleanup_error='LATE CLEANUP')
            mp.save(mp.record_path(path),row)
            report,latest=prod.retirement_supervision(context,previous)
            self.assertIn('LATE ORIGINAL',report['error'])
            self.assertEqual(report['execution_error'],"ValueError('initial workload failure')")
            self.assertEqual(report['supervision']['cleanup_error'],'LATE CLEANUP')
            self.assertEqual(latest,report['supervision'])
            self.assertNotEqual(latest['receipt_sha256'],previous['supervision']['receipt_sha256'])

    def test_unreadable_or_changed_final_receipt_refuses_without_erasing_a_known_original(self):
        for kind in ('missing','changed_nonce','partial_journal'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);path,row,context=self.fixture(root)
                row.update(state='stopping',supervision_error='KNOWN ORIGINAL');mp.save(mp.record_path(path),row)
                previous=mp.failure_report(context,ValueError('workload'))
                if kind=='missing':mp.record_path(path).unlink()
                elif kind=='changed_nonce':mp.save(mp.record_path(path),dict(row,nonce='other'))
                else:Path(row['signal_journal']).write_text('{')
                report,latest=prod.retirement_supervision(context,previous)
                self.assertIn('KNOWN ORIGINAL',report['error']);self.assertFalse(report['supervision_refresh_passed'])
                self.assertEqual(report['execution_error'],"ValueError('workload')")
                if kind!='partial_journal':
                    self.assertTrue(report['supervision']['evidence_error'])
                    self.assertEqual(report['earlier_failure'],previous)
        with tempfile.TemporaryDirectory() as d:
            _,_,context=self.fixture(Path(d));mp.record_path(Path(d)/'job.pid').unlink()
            report,_=prod.retirement_supervision(context)
            self.assertIn('unavailable after retirement',report['error'])
            self.assertFalse(report['supervision_refresh_passed'])

    def test_healthy_final_read_permits_publication_but_late_fault_or_missing_evidence_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            path,row,context=self.fixture(Path(d))
            report,evidence=prod.retirement_supervision(context)
            self.assertIsNone(report);self.assertEqual(evidence['state'],'running')
            mp.save(mp.record_path(path),dict(row,supervision_error='late fault'))
            report,_=prod.retirement_supervision(context)
            self.assertIn('late fault',report['error'])
            mp.record_path(path).unlink()
            report,_=prod.retirement_supervision(context)
            self.assertTrue(report['supervision']['evidence_error'])

    def test_worker_refreshes_after_down_and_preserves_earlier_execution_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,row,context=self.fixture(root)
            worker=object.__new__(prod.Worker);worker.root=root;worker.global_root=root;worker.supervision=context
            worker.plan={};worker.instance=3;worker.wave=0;worker.cut='a'*40;worker.session=0;worker.claim=object();worker.active_job=None
            worker.check=lambda:(_ for _ in ()).throw(ValueError('first execution failure'))
            worker.down=lambda:mp.save(mp.record_path(path),dict(row,state='stopping',supervision_error='fault during down'))
            with patch.object(prod,'backup_terminal',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'fault during down'):worker.run([{}])
            report=json.loads((root/'worker-finalization.json').read_text())
            self.assertFalse(report['passed'])
            self.assertEqual(report['supervision_failure']['execution_error'],"ValueError('first execution failure')")
            self.assertEqual(report['supervision_after_retirement']['supervision_error'],'fault during down')

    @unittest.skipUnless(sys.platform=='linux' and hasattr(os,'pidfd_open'),'Linux pidfds required')
    def test_real_supervisor_fault_during_monitor_and_backup_retirement_refreshes_final_report(self):
        # Exact review ordering, repeated: ordinary wave error -> monitor.close
        # final report -> real supervisor scan failure -> default SIGINT.
        for phase in ('monitor','monitor','backup'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as d:
                root=Path(d);path=root/'job.pid'
                worker="""import json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import triplet_production as prod
root=Path(sys.argv[1]);phase=sys.argv[2];release=root/'release.json';release.write_text('{"passed":true}')
real_time=time;clock=[0]
def monotonic():clock[0]+=120;return clock[0]
def wave(_):raise ValueError('workload failure before finalization')
real_record=prod.DiskWatch.record
def record(self):
 real_record(self)
 if self.closed and phase=='monitor':
  root.joinpath('ready').write_text('ready')
  while True:real_time.sleep(.02)
def backup(*args):
 root.joinpath('ready').write_text('ready')
 while True:real_time.sleep(.02)
original_backup=prod.backup_remaining_terminal
recipe=dict(cut='a'*40,production_root=str(root/'production'),runtime_root=str(root/'runtime'))
with patch.object(prod,'require_host'),patch.object(prod,'frozen'),patch.object(prod,'disk',return_value={}),patch.object(prod,'confined',side_effect=Path),patch.object(prod,'gpu_uuid',return_value=prod.ASSIGNED_UUID),patch.object(prod.subprocess,'check_output',return_value=''),patch.object(prod,'reduce_waves',side_effect=wave),patch.object(prod.DiskWatch,'record',record),patch.object(prod,'backup_remaining_terminal',side_effect=backup if phase=='backup' else original_backup),patch.object(prod,'time',NS(monotonic=monotonic,time_ns=real_time.time_ns,sleep=real_time.sleep)):
 try:prod.coordinate(recipe,release)
 except BaseException:pass
"""
                code="""import sys
from pathlib import Path
import managed_process as mp
root=Path(sys.argv[1]);worker=sys.argv[2];original=mp.OwnedProcesses.scan;tripped=[]
def scan(self):
 original(self)
 if root.joinpath('ready').exists() and not tripped:
  tripped.append(True);raise RuntimeError('supervisor RuntimeError during retirement')
mp.OwnedProcesses.scan=scan
mp.supervise(root/'job.pid','retirement-test',[sys.executable,'-B','-c',worker,str(root),sys.argv[3]])
"""
                env=dict(os.environ,PYTHONPATH=str(Path(mp.__file__).parent))
                with (root/'supervisor.log').open('w') as log:
                    process=subprocess.Popen([sys.executable,'-B','-c',code,str(root),worker,phase],env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
                    process.wait(timeout=20)
                supervisor=json.loads(mp.record_path(path).read_text());self.assertEqual(supervisor['state'],'failed')
                self.assertEqual(supervisor['cleanup']['remaining'],[])
                report=json.loads((root/'production/production-finalization.json').read_text())
                self.assertFalse(report['passed']);self.assertEqual(report['cleanup']['remaining'],[])
                self.assertEqual(report['backup_children_cleanup']['remaining'],[])
                failure=report['supervision_failure']
                self.assertEqual(failure['execution_error'],"ValueError('workload failure before finalization')")
                self.assertEqual(failure['supervision']['supervision_error'],supervisor['supervision_error'])
                self.assertEqual(report['supervision_after_retirement'],failure['supervision'])
                self.assertTrue(failure['supervision_refresh_passed'])
                self.assertTrue(any(r['execution_error']=='KeyboardInterrupt()' for r in report['retirement_errors']))
                faults=[json.loads(p.read_text()) for p in (root/'production').glob('fault-*.json')]
                self.assertTrue(any('supervisor RuntimeError during retirement' in json.dumps(r) for r in faults))
                print(json.dumps(dict(event='retirement_timing_regression',phase=phase,error=report['error'],execution_error=failure['execution_error'],supervision_error=failure['supervision']['supervision_error'],remaining=report['cleanup']['remaining'])),flush=True)
                self.assertFalse((root/'production/PUBLICATION-COMPLETE.json').exists())
                with self.assertRaises(mp.SupervisionFailure):mp.stop(path)
