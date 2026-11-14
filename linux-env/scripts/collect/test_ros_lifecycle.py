"""Native resources are destroyed only after their callback owners retire."""
import ast,json,os,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from ros_lifecycle import retire_executor,shutdown_context
import clock_status,collection_status,storage

class LifecycleTests(unittest.TestCase):
    def test_spin_and_queued_pool_tasks_retire_before_resource_destruction(self):
        started=threading.Event();release=threading.Event();effects=[]
        pool=ThreadPoolExecutor(1)
        first=pool.submit(lambda:(started.set(),release.wait(2),effects.append('callback retired')))
        self.assertTrue(started.wait(2));later=pool.submit(lambda:effects.append('queued callback ran'))
        spin=threading.Thread(target=lambda:None);spin.start()
        ex=NS(shutdown=lambda **kw:True,_executor=pool)
        done=threading.Event();errors=[]
        def retire():
            try:retire_executor(ex,spin,2);effects.append('resources may be destroyed')
            except BaseException as exc:errors.append(exc)
            finally:done.set()
        worker=threading.Thread(target=retire);worker.start();self.assertFalse(done.wait(.02));release.set();worker.join(3)
        self.assertFalse(errors);self.assertEqual(effects,['callback retired','resources may be destroyed']);self.assertTrue(later.cancelled())
        self.assertTrue(first.done())

    def test_false_shutdown_or_stuck_pool_refuses_retirement(self):
        spin=NS(join=lambda **kw:None,is_alive=lambda:False)
        with self.assertRaisesRegex(RuntimeError,'drain callbacks'):retire_executor(NS(shutdown=lambda **kw:False),spin,.01)
        stuck=NS(name='stalled-owner',join=lambda **kw:None,is_alive=lambda:True)
        pool=NS(shutdown=lambda **kw:None,_threads=[stuck])
        with self.assertRaisesRegex(RuntimeError,'callback pool'):retire_executor(NS(shutdown=lambda **kw:True,_executor=pool),spin,.01)

    def test_shared_node_explicitly_destroys_actions_before_base_node(self):
        root=Path(__file__).resolve().parents[3]
        path=root/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PickPlaceNode')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='destroy_node')
        effects=[]
        class Base:
            def destroy_node(self):effects.append('base node')
        class Action:
            def destroy(self):effects.append('action')
        cls.bases=[ast.Name(id='Base',ctx=ast.Load())];cls.body=[method];cls.decorator_list=[]
        scope=dict(Base=Base,ActionClient=Action);exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),scope)
        node=scope['PickPlaceNode']();node.waitables=[Action(),object(),Action()];node.destroy_node()
        self.assertEqual(effects,['action','action','base node'])

    def test_context_is_explicitly_destroyed_after_shutdown(self):
        effects=[];ctx=NS(destroy=lambda:effects.append('context destroy'))
        shutdown_context(NS(shutdown=lambda **kw:effects.append(('shutdown',kw['context'] is ctx)),get_default_context=lambda:None),ctx)
        self.assertEqual(effects,[('shutdown',True),'context destroy'])

class ClockCertificateTests(unittest.TestCase):
    def test_pass_flag_without_post_cleanup_certificate_is_not_success(self):
        with tempfile.TemporaryDirectory() as d,patch('clock_status.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path),patch('storage.confined',side_effect=Path):
            output=Path(d)/'clock.json';certificate=clock_status.begin(output)
            storage.atomic_json(output,dict(passed=True,regimes=[{},{}]))
            with self.assertRaisesRegex(RuntimeError,'incomplete'):clock_status.require(output,os.getpid(),0,0)
            good=clock_status.finish(output,certificate,dict(checks_passed=True,regimes=[{},{}]),[])
            self.assertTrue(good['passed']);self.assertTrue(clock_status.require(output,os.getpid(),0,0)['passed'])
            with self.assertRaisesRegex(RuntimeError,'abnormally'):clock_status.require(output,os.getpid(),0,-11)
            storage.atomic_json(output,dict(passed=True,regimes=[{},{}],changed=True))
            with self.assertRaisesRegex(RuntimeError,'changed'):clock_status.require(output,os.getpid(),0,0)

    def test_failed_retirement_is_certified_as_failure(self):
        with tempfile.TemporaryDirectory() as d,patch('clock_status.confined',side_effect=Path),patch('collection_status.confined',side_effect=Path),patch('storage.confined',side_effect=Path):
            output=Path(d)/'clock.json';certificate=clock_status.begin(output)
            final=clock_status.finish(output,certificate,dict(checks_passed=True,regimes=[{},{}]),[dict(stage='executor',error='still alive')])
            self.assertFalse(final['passed']);self.assertFalse(json.loads(output.read_text())['passed'])
            with self.assertRaisesRegex(RuntimeError,'failed'):clock_status.require(output,os.getpid(),0,0)
