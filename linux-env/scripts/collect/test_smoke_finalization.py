"""Runner evidence survives process exit, final receipt publication and cleanup errors."""
import ast,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock,Mock,patch
import run_smoke


class RunnerEvidenceTests(unittest.TestCase):
    def test_cleanup_exception_still_publishes_failed_receipt_and_end_marker(self):
        output=MagicMock();ledger=Mock();ledger.known={};ledger.stop.side_effect=PermissionError('injected cleanup failure');ledger.remaining.return_value=[]
        channel=MagicMock();channel.__enter__.return_value.connect_ex.return_value=111
        written=[]
        with patch('run_smoke.socket.socket',return_value=channel),patch('run_smoke.atomic_json',side_effect=lambda path,data:written.append(data)):
            result=run_smoke.finalize_smoke(output,ledger,'a'*40,True,None)
        self.assertFalse(result['passed']);self.assertIn('PermissionError',result['cleanup_error'])
        self.assertEqual(written[-1],result);output.__truediv__.return_value.write_text.assert_called_once_with('SMOKE_STOPPED\n')

    def test_instance_claim_is_released_in_outer_finally_even_if_finalizer_raises(self):
        tree=ast.parse(Path(run_smoke.__file__).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        outer=next(n for n in main.body if isinstance(n,ast.Try))
        closes=[n for n in ast.walk(ast.Module(body=outer.finalbody,type_ignores=[])) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='close']
        self.assertEqual(len(closes),1);self.assertEqual(closes[0].func.value.id,'claim')

    def test_receipt_published_at_exit_is_attributed_to_its_producing_stage(self):
        tree=ast.parse(Path(run_smoke.__file__).read_text());stage=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='stage')
        output=MagicMock();run=MagicMock();receipt=Mock();receipt.read_text.return_value='{"accepted":true}'
        receipt.stat.return_value=NS(st_mtime_ns=2*10**9);run.glob.side_effect=[[],[receipt]]
        log_path=MagicMock()
        output.__truediv__.side_effect=lambda name:run if name=='normal' else log_path
        process=Mock();process.pid=12;process.returncode=0;process.poll.side_effect=[None,0]
        times=[];stages=[];scope=dict(args=NS(cut='a'*40),output=output,env={},claim=NS(fd=77),REPO=Path('/unused'),
            ledger=Mock(),seen=set(),attempt_times=times,stages=stages,frozen=lambda cut:None,pending_receipt=run_smoke.pending_receipt,
            atomic_json=Mock(),json=json,print=lambda *a,**kw:None,time=NS(time_ns=Mock(side_effect=[10**9,3*10**9]),sleep=lambda dt:None),
            subprocess=NS(Popen=lambda *a,**kw:process,STDOUT=-2))
        exec(compile(ast.Module(body=[stage],type_ignores=[]),'actual stage','exec'),scope)
        self.assertEqual(scope['stage'](dict(name='normal-first',argv=['unused'])),0)
        self.assertEqual(len(times),1);self.assertEqual(times[0]['stage'],'normal-first');self.assertEqual(times[0]['wall_interval_s'],1.)

    def test_final_invalid_receipt_fails_even_if_its_previous_version_was_seen(self):
        tree=ast.parse(Path(run_smoke.__file__).read_text());stage=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='stage')
        output=MagicMock();run=MagicMock();receipt=Mock();receipt.read_text.side_effect=ValueError('invalid final receipt')
        run.glob.return_value=[receipt];log_path=MagicMock()
        output.__truediv__.side_effect=lambda name:run if name=='normal' else log_path
        process=Mock();process.pid=12;process.returncode=0;process.poll.return_value=0
        scope=dict(args=NS(cut='a'*40),output=output,env={},claim=NS(fd=77),REPO=Path('/unused'),ledger=Mock(),
            seen={str(receipt)},attempt_times=[],stages=[],frozen=lambda cut:None,pending_receipt=run_smoke.pending_receipt,
            atomic_json=Mock(),json=json,print=lambda *a,**kw:None,time=NS(time_ns=lambda:10**9,sleep=lambda dt:None),
            subprocess=NS(Popen=lambda *a,**kw:process,STDOUT=-2))
        exec(compile(ast.Module(body=[stage],type_ignores=[]),'actual stage','exec'),scope)
        with self.assertRaisesRegex(ValueError,'invalid final receipt'):scope['stage'](dict(name='normal-first',argv=['unused']))
        self.assertIn('invalid final receipt',scope['stages'][0]['error'])
