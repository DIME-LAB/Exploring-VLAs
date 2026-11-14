"""A failed contact pose is reset and verified before another planning request."""
import copy,threading,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"vla_SO-ARM101/src/so_arm101_control"))
from types import SimpleNamespace as NS
import numpy as np
from contract import JOINTS
from safe_home import make_baseline,validate_baseline
from so_arm101_control.grasp_geometry import _grasp_home_data_target
from test_tick_contract import checked_window
import test_restore_clock as restore_probes
from test_restore_clock import capture_snapshot
from tick_contract import TickContract,FLOAT32_DT
from dart import TargetCandidate


def baseline():
    target=_grasp_home_data_target();goal=[target[name] for name in JOINTS[:5]]+[.113]
    raw=[dict(state=goal[:],drive_targets=goal[:],sim_ns=10**9+round(i*1e9/30),physics_time_s=1.+i*4*FLOAT32_DT,
              bodies={'red':dict(position=[.2,0.,.01],orientation=[1.,0.,0.,0.],velocity=[0.]*6)}) for i in range(6)]
    samples=checked_window(raw)
    return make_baseline(dict(target=target,snapshots=samples),'source-home-episode')


class SafeHomeTests(unittest.TestCase):
    def test_baseline_rejects_missing_or_changed_evidence(self):
        value=baseline();self.assertTrue(validate_baseline(value,_grasp_home_data_target())['passed'])
        for kind in ('position','drive','goal'):
            bad=copy.deepcopy(value)
            if kind=='position':bad['snapshot']['state'][1]+=.1
            elif kind=='drive':bad['snapshot']['drive_targets'][1]+=.1
            else:bad['home_target'][1]+=.1
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):validate_baseline(bad,_grasp_home_data_target())
        with self.assertRaises(RuntimeError):validate_baseline(None,_grasp_home_data_target())

    def fixture(self):
        c,r,entry,_,_=restore_probes.RestoreHistoryTests().setup_restore();c.safe_home_baseline=baseline()
        failed=[.5]*5+[.113];r.art.set_joint_positions(np.array([failed],dtype=np.float32));r.art.get_applied_actions().joint_positions[:]=[failed]
        r.stamp=lambda:2*10**9;r.world.current_time=2.;c.tick_audit=TickContract();c.tick_audit.previous=checked_window([r.snapshot()])[0]['tick_contract']['row']
        c.servo.plan=c.servo.grip_plan=None;c.servo.reference=failed[:];c.servo.reference_epoch=1
        c.servo.previous_command=TargetCandidate(1,2*10**9,tuple(failed),(0.,)*5,(0.,)*6,(0.,)*6,tuple(failed),planner_reference=tuple(failed))
        events=[];c.workers=NS(retire=lambda:events.append('retire') or dict(passed=True))
        c.scene_reset=NS(clear=lambda:events.append('authoritative_scene_clear') or dict(verified=True))
        def restore(tool,args):
            events.append('physical_restore');return r.restore(c.safe_home_baseline['snapshot'],arm=True,restore_drives=True)
        c.mutate_kit=restore
        def advance(count):
            samples=[];start=r.stamp();world=r.world.current_time
            for i in range(1,count+1):
                snap=r.snapshot();snap.update(sim_ns=start+round(i*1e9/30),physics_time_s=world+i*4*FLOAT32_DT)
                capture_snapshot(c,r,entry,snap);samples.append(copy.deepcopy(c.last_snapshot))
            events.append('verified_ticks');return samples
        c.advance_ticks=advance
        c.sync_scene=lambda snapshot:events.append('scene_verify') or setattr(c,'scene_pose_check',dict(passed=True))
        return c,r,events

    def test_actual_restore_rebases_arm_and_drives_without_planning_from_failed_pose(self):
        c,r,events=self.fixture();result=c.restore_failed_episode_home()
        self.assertTrue(result['passed']);self.assertTrue(result['before_planning'])
        self.assertEqual(events,['retire','authoritative_scene_clear','physical_restore','verified_ticks','scene_verify'])
        np.testing.assert_allclose(r.snapshot()['state'],c.safe_home_baseline['snapshot']['state'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(r.snapshot()['drive_targets'],c.safe_home_baseline['snapshot']['drive_targets'],atol=1e-6,rtol=0)
        self.assertIsNone(c.node._attached_lego_name);self.assertIsNone(c.failure)

    def test_failed_readback_is_terminal_and_no_followup_plan_or_ticks_runs(self):
        c,r,events=self.fixture();real=c.mutate_kit
        def bad(tool,args):
            result=real(tool,args);result['snapshot']['state'][0]+=.1;return result
        c.mutate_kit=bad
        with self.assertRaises(RuntimeError):c.restore_failed_episode_home()
        self.assertNotIn('verified_ticks',events);self.assertNotIn('scene_verify',events)
        self.assertTrue(c.failure);self.assertTrue(c.generation.stalled)

    def test_mid_episode_restore_refuses(self):
        c,r,events=self.fixture();c.writer=object()
        with self.assertRaisesRegex(RuntimeError,'while recording'):c.restore_failed_episode_home()
        self.assertEqual(events,[])
