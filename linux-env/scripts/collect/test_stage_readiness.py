"""Exercise admission against transient invalid USD and the actual baseline code."""
import copy,sys,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from stage_readiness import probe_code,errors,wait


class StageFixture:
    def __init__(self):
        from types import SimpleNamespace as NS
        self.names=[c+'_2x3_'+str(i) for c in ('red','green','blue') for i in (0,1)]
        self.paths={n:'/World/Objects/'+n+'/'+n for n in self.names}
        self.paths.update({n:'/World/Containers/'+n for n in ('cup_red','cup_green','cup_blue')})
        self.missing=set();self.extra=[];self.pending=0;self.root='stage-A';self.fail_legacy=False;self.helper_calls=0
        fixture=self
        class Prim:
            def __init__(self,path,valid=True):self.path=path;self.valid=valid and path not in fixture.missing
            def __bool__(self):return self.valid
            def IsValid(self):return self.valid
            def GetName(self):return self.path.rsplit('/',1)[-1]
            def GetPath(self):return self.path
            def GetTypeName(self):return 'Material' if self.GetName()=='PhysicsMaterial' else 'Scope' if self.GetName()=='Looks' else 'Xform'
            def GetChildren(self):return [Prim('/World/Objects/'+n) for n in ['PhysicsMaterial',*fixture.names,*fixture.extra]]
        allowed={'/World/Objects','/World/SO_ARM101','/World/SO_ARM101/root_joint','/World/SO_ARM101/base_link','/World/workspace_camera',*self.paths.values()}
        stage=NS(GetPrimAtPath=lambda p:Prim(p,p in allowed),GetRootLayer=lambda:NS(identifier=self.root))
        class Xform:
            def __init__(self,p):self.p=p
            def __bool__(self):return bool(self.p)
            def ComputeLocalToWorldTransform(self,t):
                if not self.p:raise RuntimeError('Accessed schema on invalid prim')
                return NS(ExtractTranslation=lambda:[.1,.2,.3])
        def helper(name):
            self.helper_calls+=1
            if self.fail_legacy and self.helper_calls>6:return '/missing/query'
            return '/World/Objects/'+name+'/'+name
        ctx=NS(get_stage=lambda:stage,get_stage_loading_status=lambda:['' if not self.pending else 'pending.usd',0,self.pending])
        omni=types.ModuleType('omni');usd=types.ModuleType('omni.usd');usd.get_context=lambda:ctx;omni.usd=usd
        pxr=types.ModuleType('pxr');pxr.UsdGeom=NS(Xformable=Xform);pxr.Usd=NS(TimeCode=NS(Default=lambda:0))
        extension=types.ModuleType('so_arm101_dt.extension');extension.get_instance=lambda:NS(_get_prim_path=helper);extension.CUP_LAYOUT={}
        package=types.ModuleType('so_arm101_dt');package.__path__=[];package.extension=extension
        runtime=types.ModuleType('collection_runtime');runtime.runtime=lambda:NS(body_paths=self.paths)
        self.modules={'omni':omni,'omni.usd':usd,'pxr':pxr,'so_arm101_dt':package,'so_arm101_dt.extension':extension,'collection_runtime':runtime}
    def read(self):
        self.helper_calls=0;scope={}
        with patch.dict(sys.modules,self.modules):exec(probe_code(),scope)
        return dict(stage_readiness=scope['result'])


class ReadinessTests(unittest.TestCase):
    def test_exact_baseline_and_all_six_named_body_paths(self):
        f=StageFixture();proof=f.read()['stage_readiness'];self.assertEqual(errors(proof),[])
        self.assertEqual(proof['baseline_object_names'],sorted(f.names));self.assertTrue(proof['legacy_baseline_passed'])
        self.assertTrue(next(v for v in proof['rows'] if v['name']=='PhysicsMaterial')['excluded'])
    def test_transient_load_and_invalid_prim_wait_until_stable(self):
        f=StageFixture();history=[];seconds=[0.];count=[0];bad=f.paths[f.names[0]]
        def read():
            i=count[0];count[0]+=1;f.pending=1 if i==0 else 0;f.missing={bad} if i<2 else set()
            return f.read()
        report=wait(read,history.append,timeout_s=2,interval_s=.1,clock=lambda:seconds[0],pause=lambda s:seconds.__setitem__(0,seconds[0]+s))
        self.assertTrue(report['passed']);self.assertEqual(len(history),5)
        self.assertTrue(any(bad in error for error in history[0]['samples'][0]['errors']))
        self.assertFalse(history[1]['passed']);self.assertEqual(history[-1]['samples'][-1]['errors'],[])
    def test_permanent_extra_child_or_query_failure_refuses(self):
        for reason in ('extra','invalid','query'):
            with self.subTest(reason=reason):
                f=StageFixture();seconds=[0.];history=[]
                if reason=='extra':f.extra=['Looks']
                if reason=='invalid':f.missing={f.paths[f.names[0]]}
                if reason=='query':f.fail_legacy=True
                with self.assertRaisesRegex(TimeoutError,'Stage readiness deadline'):
                    wait(f.read,history.append,timeout_s=.3,interval_s=.1,clock=lambda:seconds[0],pause=lambda s:seconds.__setitem__(0,seconds[0]+s))
                self.assertFalse(any(v['passed'] for v in history))
                if reason=='extra':self.assertIn('/World/Objects/Looks/Looks',' '.join(history[-1]['samples'][-1]['errors']))
                if reason=='query':self.assertIn('/missing/query',' '.join(history[-1]['samples'][-1]['errors']))
    def test_root_changes_reset_stability_and_late_reply_never_admits(self):
        f=StageFixture();seconds=[0.];history=[];count=[0]
        def read():
            f.root='A' if count[0]==0 else 'B';count[0]+=1;return f.read()
        self.assertTrue(wait(read,history.append,timeout_s=2,interval_s=.1,clock=lambda:seconds[0],pause=lambda s:seconds.__setitem__(0,seconds[0]+s))['passed'])
        self.assertEqual(len(history),4)
        seconds=[0.];history=[]
        def late():seconds[0]+=2;return f.read()
        with self.assertRaises(TimeoutError):wait(late,history.append,timeout_s=1,stable_samples=1,clock=lambda:seconds[0],pause=lambda s:None)
        self.assertFalse(history[-1]['passed'])
    def test_actual_attempt_never_spawns_collector_before_admission(self):
        from regrasp_runner import Processes
        p=object.__new__(Processes);p.capacity=lambda:None;p.root=lambda c:Path('/not-written')
        p.config={'instance':3};p.wrapper='not-executed';p.stage=Mock();p.observe=Mock()
        p.wait_readiness=Mock(side_effect=TimeoutError('invalid named prim'))
        with self.assertRaisesRegex(TimeoutError,'invalid named prim'):p.attempt(dict(base_seed=600000),0,600000)
        p.stage.assert_not_called();p.observe.assert_not_called()


if __name__=='__main__':unittest.main()
