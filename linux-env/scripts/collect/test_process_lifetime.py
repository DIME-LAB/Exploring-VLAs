"""Real Linux pidfds, long-run child churn and detached-launch signal isolation."""
import fcntl,json,os,select,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import managed_process
from run_smoke import OwnedProcesses,identity,pidfd_ready


@unittest.skipUnless(sys.platform=='linux' and hasattr(os,'pidfd_open'),'Linux pidfds required')
class ProcessLifetimeTests(unittest.TestCase):
    def wait_for(self,predicate,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=predicate()
            if value:return value
            time.sleep(.02)
        self.fail('Condition did not become true before deadline')

    def test_pidfd_above_select_ceiling_stays_birth_bound_and_signals_owned_child(self):
        process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        ledger=OwnedProcesses(Path('/unused'),0)
        try:
            low=os.pidfd_open(process.pid);high=fcntl.fcntl(low,fcntl.F_DUPFD_CLOEXEC,2048);os.close(low)
            with patch('run_smoke.os.pidfd_open',return_value=high):ledger.add(process.pid)
            self.assertEqual(ledger.handles[process.pid],high)
            with self.assertRaises(ValueError):select.select([high],[],[],0)
            self.assertFalse(pidfd_ready(high))
            ledger.send(process.pid,signal.SIGTERM);process.wait(timeout=5)
            self.assertTrue(pidfd_ready(high));self.assertEqual(ledger.stop()['remaining'],[])
        finally:
            if process.poll() is None:process.kill();process.wait()
            for fd in ledger.handles.values():os.close(fd)

    def test_1200_short_lived_children_retire_handles_but_keep_all_receipts(self):
        ledger=OwnedProcesses(Path('/unused'),0);peak=0
        try:
            for _ in range(1200):
                child=subprocess.Popen(['/bin/cat'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL)
                ledger.add(child.pid);peak=max(peak,len(ledger.handles));child.communicate();ledger.scan()
            self.assertLessEqual(peak,1);self.assertEqual(ledger.handles,{})
            self.assertEqual(len(ledger.tracked()),1200)
            self.assertEqual(ledger.stop()['remaining'],[])
        finally:
            for fd in ledger.handles.values():os.close(fd)

    def test_pid_reuse_requires_new_ancestry_and_kernel_handle_preserves_old_receipt(self):
        ledger=OwnedProcesses(Path('/unused'),0)
        old=dict(pid=123,start_ticks='old',ppid=ledger.root['pid'])
        new=dict(old,start_ticks='new')
        ledger.known[123]=old
        with patch('run_smoke.identity',side_effect=lambda pid:new if pid==123 else ledger.root),\
             patch('run_smoke.os.pidfd_open',return_value=77),patch('run_smoke.pidfd_ready',return_value=False):
            ledger.add(123)
        self.assertEqual(ledger.retired,[old]);self.assertEqual(ledger.known[123]['start_ticks'],'new')
        self.assertEqual(len(ledger.tracked()),2)

    def test_launch_shell_hup_and_int_do_not_reach_detached_supervisor_or_workload(self):
        for sig in (signal.SIGHUP,signal.SIGINT):
            with self.subTest(signal=sig),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/'job.pid';receipt=managed_process.record_path(path)
                code="""import sys,time,json
from pathlib import Path
import managed_process
root=Path(sys.argv[1])
managed_process.start(root/'job.pid',root/'job.log',[sys.executable,'-c','import time;time.sleep(120)'])
(root/'ready').write_text('ready')
time.sleep(120)
"""
                env=dict(os.environ,PYTHONPATH=str(Path(managed_process.__file__).parent))
                launch=subprocess.Popen([sys.executable,'-c',code,str(root)],env=env,start_new_session=True,
                    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                row=None
                try:
                    self.wait_for(lambda:(root/'ready').exists());row=json.loads(receipt.read_text())
                    launch_id=identity(launch.pid)
                    for key in ('supervisor','workload'):
                        current=identity(row[key]['pid'])
                        self.assertEqual(current['sid'],current['pid'])
                        self.assertNotEqual(current['sid'],launch_id['sid'])
                    # Signal only the freshly spawned launching shell's group.
                    self.assertEqual(identity(launch.pid)['start_ticks'],launch_id['start_ticks'])
                    os.killpg(launch.pid,sig);launch.wait(timeout=5)
                    for key in ('supervisor','workload'):
                        current=identity(row[key]['pid'])
                        self.assertEqual(current['start_ticks'],row[key]['start_ticks'])
                        self.assertNotEqual(current['state'],'Z')
                    result=managed_process.stop(path)
                    self.assertTrue(result['stopped']);self.assertEqual(result['cleanup']['remaining'],[])
                    final=json.loads(receipt.read_text());self.assertEqual(final['stop_signals'],['requested'])
                    events=[json.loads(line) for line in Path(final['signal_journal']).read_text().splitlines()]
                    self.assertTrue(events)
                    self.assertTrue(all(x['sender']['pid']==row['supervisor']['pid'] for x in events))
                finally:
                    if row and managed_process.verified(path):managed_process.stop(path)
                    if launch.poll() is None:launch.terminate();launch.wait(timeout=5)

    def test_supervision_failure_records_original_cause_before_own_cleanup_sigint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=dict(os.environ,PYTHONPATH=str(Path(managed_process.__file__).parent))
            # Reproduce the original select(high pidfd) failure once in scan.
            # Cleanup must persist its cause and explicitly attribute its SIGINT.
            code="""import os,sys,time,fcntl,select
from pathlib import Path
import managed_process
root=Path(sys.argv[1]);original=managed_process.OwnedProcesses.scan;tripped=[]
def scan(self):
 original(self)
 if (root/'worker-ready').exists() and not tripped:
  tripped.append(True)
  low=os.open('/dev/null',os.O_RDONLY);high=fcntl.fcntl(low,fcntl.F_DUPFD_CLOEXEC,2048)
  try:select.select([high],[],[],0)
  finally:os.close(low);os.close(high)
managed_process.OwnedProcesses.scan=scan
worker='import signal,time;from pathlib import Path;root=Path('+repr(str(root))+');signal.signal(signal.SIGINT,lambda s,f:(root.joinpath("received-signal").write_text(str(s)),exit(0)));root.joinpath("worker-ready").write_text("ready");time.sleep(120)'
managed_process.supervise(root/'job.pid','test-failure',[sys.executable,'-c',worker])
"""
            log=root/'probe.log'
            with log.open('w') as stream:
                process=subprocess.Popen([sys.executable,'-c',code,str(root)],env=env,start_new_session=True,
                    stdout=stream,stderr=subprocess.STDOUT)
                process.wait(timeout=15)
            self.assertNotEqual(process.returncode,0)
            row=json.loads(managed_process.record_path(root/'job.pid').read_text())
            self.assertEqual(row['state'],'failed');self.assertIn('filedescriptor out of range',row['supervision_error'])
            self.assertEqual(row['cleanup']['remaining'],[])
            self.assertEqual((root/'received-signal').read_text(),'2')
            event=json.loads(Path(row['signal_journal']).read_text().splitlines()[0])
            self.assertEqual(event['sender']['pid'],process.pid)
            self.assertEqual(event['target']['pid'],row['workload']['pid'])
            self.assertEqual(event['signal'],2)

    def test_closed_pidfd_is_refused_instead_of_treated_as_live(self):
        fd=os.open('/dev/null',os.O_RDONLY);os.close(fd)
        with self.assertRaises(OSError):pidfd_ready(fd)

    def test_supervision_and_cleanup_failures_both_survive_in_final_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'job.pid'
            from unittest.mock import Mock
            ledger=Mock();ledger.scan.side_effect=RuntimeError('original scan failure')
            ledger.stop.side_effect=RuntimeError('cleanup failure');ledger.tracked.return_value=[]
            worker=Mock();worker.pid=123;worker.poll.return_value=None
            with patch('managed_process.OwnedProcesses',return_value=ledger),\
                 patch('managed_process.signal.signal'),\
                 patch('managed_process.subprocess.Popen',return_value=worker):
                with self.assertRaisesRegex(RuntimeError,'original scan failure'):
                    managed_process.supervise(path,'test-both-failures',['unused'])
            final=json.loads(managed_process.record_path(path).read_text())
            self.assertEqual(final['state'],'cleanup_failed')
            self.assertIn('original scan failure',final['supervision_error'])
            self.assertIn('cleanup failure',final['cleanup_error'])
            self.assertTrue(final['cleanup']['remaining'])
            with self.assertRaisesRegex(RuntimeError,'Unresolved cleanup: unverified descendant'):
                managed_process.verified(path)

    def test_exited_handle_cannot_close_while_an_owned_signal_is_in_flight(self):
        import threading
        child=subprocess.Popen(['/bin/sleep','30'],start_new_session=True)
        ledger=OwnedProcesses(Path('/unused'),0);ledger.add(child.pid)
        entered=threading.Event();release=threading.Event();scanning=threading.Event();scanned=threading.Event();errors=[]
        def before(event):entered.set();release.wait(5)
        def send():
            try:ledger.send(child.pid,signal.SIGINT)
            except BaseException as exc:errors.append(exc)
        def scan():
            scanning.set()
            try:ledger.scan()
            except BaseException as exc:errors.append(exc)
            finally:scanned.set()
        ledger.before_signal=before;sender=threading.Thread(target=send);scanner=threading.Thread(target=scan)
        try:
            sender.start();self.assertTrue(entered.wait(5))
            child.terminate();child.wait(timeout=5);scanner.start();self.assertTrue(scanning.wait(5))
            self.assertFalse(scanned.wait(.05));fcntl.fcntl(ledger.handles[child.pid],fcntl.F_GETFD)
            release.set();sender.join(5);scanner.join(5)
            self.assertFalse(sender.is_alive());self.assertFalse(scanner.is_alive());self.assertEqual(errors,[])
            self.assertEqual(ledger.handles,{})
        finally:
            release.set()
            if sender.ident:sender.join(5)
            if scanner.ident:scanner.join(5)
            if child.poll() is None:child.kill();child.wait()
            for fd in ledger.handles.values():os.close(fd)
