"""Real Collector/TripletSession CPU integration; only ROS/Kit backends are fakes.

Run on dual with Humble sourced, CUDA hidden, and TMPDIR on owned backup storage.
Recorded native responses are fixtures, never live-physics qualification evidence.
"""
import argparse,ast,copy,json,os,sys,tempfile,threading,types,unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

REPO=Path('/mnt/a4500-backup/exploring-vlas/eval-native-triplet360')
BASE=Path('/mnt/a4500-backup/exploring-vlas')
sys.dont_write_bytecode=True
sys.path[:0]=[str(REPO/'linux-env/scripts/eval'),str(REPO/'linux-env/scripts/collect')]
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import collect as c
import target_triplets as producer
import triplet_reset as reset
from triplet_bank import load,document


class Variable:
    def set(self,value):self.value=value


class Node:
    def __init__(self,**kwargs):
        self.context=object();self.validity_client=object();self.js_sub=object()
        self.objects_lock=threading.RLock();self._drop_lock=threading.RLock()
        self.objects_data={};self._drop_data={};self._qs_abort_evt=threading.Event()
        self._motion_thread=lambda *a,**k:None;self._motion_token=lambda *a,**k:nullcontext()
        self._grasp_approach_height_var=Variable();self._cup_collision_names=[];self._lego_collision_names=[]
    def destroy_subscription(self,*args):pass
    def create_publisher(self,*args):return object()
    def get_clock(self):return types.SimpleNamespace(now=lambda:types.SimpleNamespace(nanoseconds=0))
    def _cmd_toggle_ground_plane(self):pass
    def _acm_allow_ground_rest(self,names):pass
    def _motion_guard(self):return nullcontext()


class Generation:
    def __init__(self):self.epoch=1
    def own(self,*args):return nullcontext()
    def require(self,epoch):assert epoch==self.epoch


class Thread:
    def __init__(self,*args,**kwargs):pass
    def start(self):pass


class Executor:
    def __init__(self,**kwargs):pass
    def add_node(self,node):pass
    def spin(self):pass


class Marker:
    def __init__(self,*args):pass
    def require_fresh_stack(self):pass


class Sensors:
    def __init__(self,*args):self.duplicate_messages={'top':0,'wrist':0}


class Servo:
    def __init__(self,*args):self.reference=None;self.reference_epoch=None
    def configure(self,params):self.params=copy.deepcopy(params)
    def execute(self,*args):pass
    def gripper(self,*args):pass


class Workers:
    def __init__(self,*args):pass
    def thread(self,*args,**kwargs):pass


class Scene:
    """Fake MoveIt service boundary; Collector.scene/sync_scene remain REAL."""
    def __init__(self,node,messages,srdf,**kwargs):
        self.node=node;self.msg=messages;self.primitive_cups=True;self.clear_reports=[]
    def clear(self):
        result=dict(passed=True,fixture_only=True,sequence=len(self.clear_reports)+1)
        self.clear_reports.append(result);return result
    def diff(self):
        from moveit_msgs.msg import PlanningScene
        return PlanningScene()
    def apply(self,scene):return types.SimpleNamespace(success=True)
    def rebuild_legos(self,snapshot):self.node._lego_collision_names=[n for n in snapshot['bodies'] if '_2x3_' in n]
    def set_policy(self,*args):pass
    def verify(self,*args):return dict(passed=True,fixture_only=True)


class Kit:
    """Clocked recorded native response fixtures; no simulator calls."""
    def __init__(self,collector,source,anchor):self.c=collector;self.source=source;self.anchor=anchor;self.restores=0;self.ticks=[];self.current=copy.deepcopy(source['episode_start_physical'])
    def timeline(self,play):assert play is False
    def restore(self,code):
        tree=ast.parse(code);call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='restore')
        requested=ast.literal_eval(call.args[0]);assert 'state' in requested
        self.restores+=1
        expected=self.source['initial_physical'] if self.restores==1 else self.anchor
        if requested!=expected:raise AssertionError('Native restore payload differs from recorded scene/anchor')
        expected_kwargs={'arm':True} if self.restores==1 else dict(arm=True,restore_drives=True,triplet_state=True)
        if {k.arg:ast.literal_eval(k.value) for k in call.keywords}!=expected_kwargs:raise AssertionError('Native restore application mode differs')
        if self.restores==1:r=dict(snapshot=copy.deepcopy(self.source['initial_physical']))
        elif self.restores==2:r=copy.deepcopy(self.source['counterfactual']['priming_restore'])
        elif self.restores==3:r=dict(copy.deepcopy(self.source['initial_restore_check']),snapshot=copy.deepcopy(self.source['initial_physical']))
        else:raise AssertionError('Unexpected additional native restore')
        self.current=copy.deepcopy(r['snapshot']);self.c.generation.epoch=self.current['epoch'];return r
    def advance(self,count):
        self.ticks.append(count)
        if count==1:self.current=copy.deepcopy(self.source['episode_start_physical'])
        self.c.generation.epoch=self.current['epoch'];return [copy.deepcopy(self.current) for _ in range(count)]
    def mutate(self,tool,params):
        assert tool=='run_python' and '_is_in_camera_view' in params['code']
        return copy.deepcopy(self.source['counterfactual']['visibility'])


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,_=load(BASE/'eval/triplet360/held20-v2','0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d')
        assert len(cls.rows)==60 and not c.rclpy.ok() and 'torch' not in sys.modules

    def exercise(self,row,output,old=False):
        source=document(row['source_receipt'],row['source_receipt_sha256'])
        patches=[patch('so_arm101_control.pick_place.PickPlaceNode',Node),patch.object(c,'SceneSafetyMarker',Marker),patch.object(c,'StepGeneration',Generation),patch.object(c,'StrictQueryClient',lambda client,*a,**k:client),patch.object(c,'Sensors',Sensors),patch.object(c,'Servo',Servo),patch.object(c,'publisher_executor_type',lambda:Executor),patch.object(c.threading,'Thread',Thread),patch.object(c,'SceneReconciler',Scene),patch.object(c,'WorkerRegistry',Workers),patch.object(c,'mcp',lambda *a,**k:True)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:stack.enter_context(p)
            args=argparse.Namespace(resume=False,output=output,replay_receipt=None,wait_rtf_floor=.1)
            collector=c.Collector(args)  # The actual complete production constructor.
            backend=Kit(collector,source,document(row['anchor_path'],row['anchor_sha256'])['anchor'])
            collector.timeline=backend.timeline;collector.restore_physical=backend.restore;collector.advance_ticks=backend.advance;collector.mutate_kit=backend.mutate
            collector.gpu_footprint=lambda:dict(passed=True,fixture_only=True)
            collector.camera_graph=lambda:dict(passed=True,fixture_only=True)
            collector.generation.epoch=source['episode_start_physical']['epoch']
            # Same production initialization effect as _initialize_scene_session:
            # its real sync_scene sets scene_pose_check but NOT scene_clear_check.
            collector.episode_start_physical=copy.deepcopy(source['episode_start_physical']);collector.sync_scene()
            self.assertFalse(hasattr(collector,'scene_clear_check'))
            collector.scene_reset.clear_reports.clear()
            receipt=dict(episode_id='cpu-'+str(row['seed']),params=copy.deepcopy(source['params']))
            path=output/'attempt-0000';path.mkdir();epochs=set()
            if old:
                receipt['params']=reset.install_recorded_session(collector,row,output)
                # Without replay initialization there are only the producer's
                # two restores, so provide those fixture responses in order.
                backend.restores=1
                collector._prepare_scene_attempt(receipt['params'],{},receipt,epochs,output,path)
            else:
                reset.prepare_recorded_scene(collector,row,{},receipt,epochs,output,path)
                self.assertEqual(backend.ticks,[2,20,1]);self.assertEqual(backend.restores,3)
                self.assertIsNone(collector.args.replay_receipt)
                self.assertEqual(receipt['planning_scene_clear_check'],collector.scene_reset.clear_reports[0])
                self.assertEqual(receipt['counterfactual']['sampler_invocations'],0)
                self.assertEqual(receipt['episode_start_physical'],source['episode_start_physical'])
                self.assertEqual(receipt['params']['object'],source['params']['object'])
                self.assertEqual(receipt['eval_scene_initialization']['protocol'],'collector-scene-recorded-replay-v1')

    def test_old_path_reproduces_missing_scene_clear_check(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SOARM_INSTANCE_ROOT':directory}):
            with self.assertRaisesRegex(AttributeError,'scene_clear_check'):self.exercise(self.rows[0],Path(directory),old=True)

    def test_all60_through_real_constructor_replay_scene_and_prepare(self):
        for row in self.rows:
            with self.subTest(case=row['case_id']),tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SOARM_INSTANCE_ROOT':directory}):
                self.exercise(row,Path(directory))
        self.assertFalse(c.rclpy.ok());self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
