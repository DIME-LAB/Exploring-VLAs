"""Frozen smoke recipe and exact-process cleanup; no simulator launches."""
import unittest
from pathlib import Path
from unittest.mock import patch
import run_smoke


class SmokeRunnerTests(unittest.TestCase):
    def test_recipe_has_normal_and_dedicated_regrasp_with_diagnostic_retention(self):
        recipe=run_smoke.plan('/mnt/a4500-backup/exploring-vlas/datasets/unit-future-smoke-not-created','a'*40)
        self.assertEqual(recipe['domain'],90);self.assertEqual(recipe['bridge_port'],8777)
        commands=[s['argv'] for s in recipe['stages'] if '-first' in s['name'] or '-resume' in s['name']]
        self.assertEqual(len(commands),4)
        self.assertTrue(all('--diagnostic' in c and c[c.index('--episodes')+1]=='5' for c in commands))
        self.assertEqual(sum('--regrasp-only' in c for c in commands),2)
        self.assertFalse(recipe['pilot_started'])

    def test_foreign_output_or_nonassigned_instance_refuses(self):
        with self.assertRaises(ValueError):run_smoke.plan('/tmp/not-authorized','a'*40)
        with self.assertRaises(ValueError):run_smoke.plan('/mnt/a4500-backup/exploring-vlas/datasets/unit-unused-smoke','a'*40,1)

    def test_source_change_refuses_before_any_stage(self):
        with patch('run_smoke.subprocess.check_output',side_effect=['collect.py\n','']):
            with self.assertRaisesRegex(RuntimeError,'Frozen smoke source'):run_smoke.frozen('a'*40)

    def test_reused_pid_never_authorizes_a_signal(self):
        owned=run_smoke.OwnedProcesses(Path('/not-used'),0)
        owned.known[123]=dict(pid=123,start_ticks='old')
        newer=dict(pid=123,start_ticks='new',ppid=1,state='S')
        with patch('run_smoke.identity',return_value=newer),patch('run_smoke.os.kill') as kill:
            self.assertEqual(owned.remaining(),[])
            owned.stop();kill.assert_not_called()

    def test_startup_traceback_is_primary_when_no_attempt_receipt_exists(self):
        from unittest.mock import Mock
        run=Mock();path=Mock();path.exists.return_value=False;run.__truediv__=Mock(return_value=path)
        log=Mock();log.read_text.return_value='Traceback (most recent call last):\n  Collector(args)\nCancelledGeneration: original startup cause\n'
        with self.assertRaises(run_smoke.StartupFailure) as failure:
            run_smoke.first_receipt(run,'normal-first',1,log)
        self.assertIn('CancelledGeneration: original startup cause',str(failure.exception))
        self.assertNotIn('FileNotFoundError',str(failure.exception))
        self.assertEqual(failure.exception.diagnostic['attempts'],0)
        self.assertEqual(failure.exception.diagnostic['traceback'],log.read_text.return_value)

    def test_foreign_pidfile_is_never_an_ownership_source(self):
        owned=run_smoke.OwnedProcesses(Path('/foreign-runtime'),0)
        owned.root=dict(pid=999,start_ticks='root')
        with patch('run_smoke.identity',return_value=None):owned.scan()
        self.assertEqual(owned.known,{})

    def test_reuse_immediately_before_signal_is_refused(self):
        owned=run_smoke.OwnedProcesses(Path('/unused'),0)
        owned.known[123]=dict(pid=123,start_ticks='old');owned.handles[123]=77
        with patch('run_smoke.identity',return_value=dict(pid=123,start_ticks='new')),\
             patch('run_smoke.signal.pidfd_send_signal',create=True) as send:
            owned.send(123,2);send.assert_not_called()

    def test_signal_uses_bound_pidfd_even_if_pid_reused_after_last_check(self):
        owned=run_smoke.OwnedProcesses(Path('/unused'),0)
        owned.known[123]=dict(pid=123,start_ticks='old');owned.handles[123]=77
        with patch('run_smoke.identity',return_value=dict(pid=123,start_ticks='old')),\
             patch('run_smoke.select.select',return_value=([],[],[])),\
             patch('run_smoke.signal.pidfd_send_signal',create=True) as send,patch('run_smoke.os.kill') as numeric:
            owned.send(123,2);send.assert_called_once_with(77,2);numeric.assert_not_called()

    def test_ancestry_is_required_before_registration(self):
        owned=run_smoke.OwnedProcesses(Path('/unused'),0);owned.root=dict(pid=999,start_ticks='root')
        foreign=dict(pid=123,start_ticks='foreign',ppid=1)
        with patch('run_smoke.os.pidfd_open',return_value=77,create=True),\
             patch('run_smoke.identity',side_effect=[foreign,owned.root]),patch('run_smoke.os.close') as close:
            owned.add(123);self.assertEqual(owned.known,{});close.assert_called_once_with(77)

    def test_overlapping_claim_refuses_and_closes_its_descriptor(self):
        with patch('run_smoke.os.open',return_value=77),\
             patch('run_smoke.fcntl.flock',side_effect=BlockingIOError),patch('run_smoke.os.close') as close:
            with self.assertRaisesRegex(RuntimeError,'already claimed'):run_smoke.InstanceClaim(Path('/unused'))
            close.assert_called_once_with(77)

    def test_live_reader_defers_partial_and_missing_then_reads_complete_receipt(self):
        from unittest.mock import Mock
        path=Mock();path.read_text.side_effect=['',FileNotFoundError(),'{"accepted":true}']
        self.assertIsNone(run_smoke.pending_receipt(path))
        self.assertIsNone(run_smoke.pending_receipt(path))
        self.assertEqual(run_smoke.pending_receipt(path),dict(accepted=True))

    def test_final_invalid_receipt_is_terminal_and_keeps_collector_log(self):
        from unittest.mock import MagicMock,Mock
        run=MagicMock();path=run.__truediv__.return_value;path.exists.return_value=True;path.read_text.return_value='{"accepted":'
        log=Mock();log.read_text.return_value='original collector error'
        with self.assertRaisesRegex(RuntimeError,'Final attempt receipt is invalid') as failure:
            run_smoke.first_receipt(run,'normal-first',1,log)
        self.assertIn('original collector error',str(failure.exception))

    def test_final_escalation_signals_an_orphan_discovered_after_its_parent_exits(self):
        from types import SimpleNamespace as NS
        owned=run_smoke.OwnedProcesses(Path('/unused'),0)
        parent=dict(pid=11,start_ticks='parent',ppid=1,pgid=11,state='S')
        child=dict(pid=12,start_ticks='child',ppid=1,pgid=777,state='S')
        live={11:True,12:False};clock=[0.];signals=[]
        owned.known={11:parent};owned.handles={11:77}
        def scan():
            if live[12]:owned.known[12]=child;owned.handles[12]=78
        owned.scan=scan
        def send(fd,sig):
            signals.append((fd,sig))
            if sig==run_smoke.signal.SIGKILL:
                if fd==77:live[11]=False;live[12]=True
                elif fd==78:live[12]=False
        def now():clock[0]+=.01;return clock[0]
        with patch('run_smoke.identity',side_effect=lambda pid:({11:parent,12:child}[pid] if live.get(pid) else None)),\
             patch('run_smoke.signal.pidfd_send_signal',side_effect=send,create=True),\
             patch('run_smoke.select.select',return_value=([],[],[])),patch('run_smoke.os.close'),\
             patch('run_smoke.time',NS(monotonic=now,sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))):
            result=owned.stop()
        self.assertEqual(result['remaining'],[])
        self.assertIn((78,run_smoke.signal.SIGKILL),signals)
