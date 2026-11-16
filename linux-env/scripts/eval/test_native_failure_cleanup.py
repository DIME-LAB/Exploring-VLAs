"""Execute the actual failed-rollout handler and cleanup with owned CPU children."""
from pathlib import Path
import ast,copy,json,subprocess,sys,tempfile,types,unittest

class PrimaryFailure(RuntimeError):pass

class CleanupTests(unittest.TestCase):
    def test_primary_rollout_error_survives_trace_writer_and_owned_cleanup(self):
        source=Path(__file__).with_name('native_policy.py')
        tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        outer=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
        def rollout(n):
            return isinstance(n,ast.Try) and any(isinstance(v,ast.Assign) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=='run_actor' and v.value.args and isinstance(v.value.args[0],ast.Attribute) and v.value.args[0].attr=='name' for v in n.body)
        handler=next(n for n in ast.walk(outer) if rollout(n))
        program=ast.fix_missing_locations(ast.Module(body=[ast.Try(body=[copy.deepcopy(handler)],handlers=[],orelse=[],finalbody=copy.deepcopy(outer.finalbody))],type_ignores=[]))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'ep00';out.mkdir();key=root/'rpc.key';key.write_bytes(b'test-only-key')
            actor_process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
            server_process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
            closed=[]
            class Closer:
                def __init__(self,name):self.name=name
                def close(self):closed.append(self.name)
            class Actor:
                def close(self):actor_process.terminate();actor_process.wait(timeout=5);closed.append('actor')
            def terminate(process):process.terminate();process.wait(timeout=5);closed.append('server')
            def fail(*args,**kwargs):raise PrimaryFailure('forced-primary-rollout-error')
            log=(root/'server.log').open('w')
            env=dict(json=json,recording=True,mcp=lambda *a:closed.append('recorder'),out=out,color='blue',row=dict(seed=900002,task='Pick a blue lego and place it in blue cup'),a=types.SimpleNamespace(out=root,max_actions=1350,evaluation_kind='TEST',weights_sha256='weight',bank_sha256='bank'),run_actor=fail,active=dict(physical=[dict(partial=True)],commands=[dict(partial=True)],captures=[]),actor=Actor(),server=server_process,terminate=terminate,serverlog=log,connection=Closer('connection'),listener=Closer('listener'),transport=Closer('transport'),rpc_key=key,key=key)
            try:
                with self.assertRaisesRegex(PrimaryFailure,'forced-primary-rollout-error'):
                    exec(compile(program,str(source),'exec'),env)
                self.assertEqual(json.loads((out/'result.json').read_text())['error'],'PrimaryFailure: forced-primary-rollout-error')
                self.assertFalse(key.exists());self.assertTrue(log.closed)
                self.assertEqual(set(closed),{'recorder','actor','server','connection','listener','transport'})
                self.assertEqual(json.loads((root/'process-cleanup.json').read_text())['errors'],[])
                self.assertIsNotNone(actor_process.poll());self.assertIsNotNone(server_process.poll())
                self.assertEqual(json.loads((out/'physical.jsonl').read_text()),dict(partial=True))
            finally:
                log.close()
                for process in (actor_process,server_process):
                    if process.poll() is None:process.terminate()
                    process.wait(timeout=5)

if __name__=='__main__':unittest.main()
