"""Native restore/capture history and preserved independent simulator clocks."""
import copy,json,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from dart import TargetCandidate
from contract import JOINTS
from tick_contract import TickContract,TickContractError,require_snapshots,observation_errors,FLOAT32_DT,STEP_NS,world_step_errors
from test_tick_contract import checked_window
import test_review_regressions as probes

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('retained_clock_pairs.json').read_text())


def capture_snapshot(c,r,entry,snapshot):
    stamp=snapshot['sim_ns'];entry['tick']=(copy.deepcopy(snapshot),time.monotonic_ns())
    for key in ('joint','wrist','top'):
        msg=entry[key][0];msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(stamp,10**9)
        entry[key]=(msg,time.monotonic_ns())
    entry['joint'][0].position=list(snapshot['state']);entry['joint'][0].velocity=list(snapshot['velocity'])
    r.world.current_time=snapshot['physics_time_s'];r.stamp=lambda:stamp
    c.node.get_clock=lambda:NS(now=lambda:NS(nanoseconds=stamp))
    r.ack_pub=NS(publish=lambda msg:c.sensors.acks.update({(json.loads(msg.data)['epoch'],json.loads(msg.data)['observation_stamp_ns']):json.loads(msg.data)}))
    c.sensors.read_ack=lambda epoch,t:c.sensors.acks[(epoch,t)]
    def read(**kwargs):c.stop.set();return stamp,entry
    c.sensors.read=read;c.stop.clear();c.capture()


class RestoreHistoryTests(unittest.TestCase):
    def setup_restore(self):
        c,r,entry=probes.ReviewRegressions().collector();v=c.servo
        v.plan=v.grip_plan=None;goal=[.1]*5+[0.];measured=[.099]*5+[0.]
        v.reference=goal[:];v.reference_epoch=1
        v.previous_command=TargetCandidate(1,10**9,tuple(goal),(0.,)*5,(0.,)*6,(0.,)*6,tuple(goal),planner_reference=tuple(goal))
        r.art.get_applied_actions().joint_positions[:]=[goal]
        state=np.array([measured],dtype=np.float32);velocity=np.zeros((1,6),dtype=np.float32)
        r.art.set_joint_positions=lambda value,**kwargs:state.__setitem__(slice(None),value)
        r.art.set_joint_velocities=lambda value,**kwargs:velocity.__setitem__(slice(None),value)
        body=dict(position=[.2,0.,.01],orientation=[1.,0.,0.,0.],velocity=[0.]*6)
        r.bodies={'red':NS(set_world_poses=lambda *a:None,set_velocities=lambda *a:None)}
        r.require_scene=lambda:None
        def snapshot():return dict(epoch=r.epoch,sim_ns=r.stamp(),physics_time_s=r.world.current_time,
            state=state[0].tolist(),velocity=velocity[0].tolist(),drive_targets=r.art.get_applied_actions().joint_positions[0].tolist(),
            bodies={'red':copy.deepcopy(body)},state_source='isaac_articulation',local_corners={})
        r.snapshot=snapshot
        old=checked_window([snapshot()])[0]['tick_contract']['row']
        old.update(action=goal[:],expert_reference=goal[:],executed_action=goal[:],applied_command=goal[:],
                   planner_execution_reference=goal[:],physics_drive_targets=goal[:],command_velocity=[0.]*5)
        c.tick_audit=TickContract();c.tick_audit.previous=old
        initial=snapshot();c.mutate_kit=lambda tool,args:r.restore(initial,arm=True)
        return c,r,entry,initial,goal

    def test_actual_restore_then_six_unrecorded_ticks_preserve_clean_home_and_rebase_targets(self):
        from success import initial_arm
        c,r,entry,initial,goal=self.setup_restore()
        restored=c.restore_physical('actual Runtime.restore through private adapter')
        self.assertEqual(c.servo.reference,goal);self.assertEqual(c.servo.reference_epoch,2)
        self.assertEqual(list(c.servo.previous_command.positions),restored['snapshot']['drive_targets'])
        self.assertEqual(c.tick_audit.boundary['kind'],'restore')
        snapshots=[];world0=restored['snapshot']['physics_time_s']
        for i in range(1,7):
            s=r.snapshot();s.update(sim_ns=10**9+round(i*1e9/30),physics_time_s=world0+i*4*FLOAT32_DT)
            capture_snapshot(c,r,entry,s)
            self.assertIsNone(c.failure);snapshots.append(copy.deepcopy(c.last_snapshot))
        self.assertAlmostEqual(snapshots[0]['tick_contract']['row']['command_velocity'][0],.03,places=5)
        self.assertTrue(require_snapshots(snapshots)['passed'])
        self.assertTrue(initial_arm(snapshots,goal[:5])['passed'])

    def test_restore_receipt_is_recognized_even_at_the_direct_cancel_boundary(self):
        c,r,entry,initial,goal=self.setup_restore()
        restored=c.cancel_owner(lambda:r.restore(initial,arm=True))
        self.assertEqual(c.tick_audit.boundary['kind'],'restore')
        self.assertEqual(list(c.servo.previous_command.positions),restored['command_baseline']['drive_targets'])
        s=r.snapshot();s.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
        capture_snapshot(c,r,entry,s);self.assertIsNone(c.failure)

    def test_restore_target_overwrite_before_first_tick_is_terminal(self):
        c,r,entry,_,_=self.setup_restore();c.restore_physical('restore')
        r.art.get_applied_actions().joint_positions[0,0]+=.02
        s=r.snapshot();s.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
        before=len(c.events);capture_snapshot(c,r,entry,s)
        self.assertTrue(c.failure);self.assertIn('boundary baseline',c.failure)
        self.assertEqual(c.events[before:],['pause'])

    def test_pause_preserves_targets_and_is_distinct_from_arm_restore(self):
        c,r,entry,_,goal=self.setup_restore();c.timeline(False)
        self.assertEqual(c.tick_audit.boundary['kind'],'pause')
        self.assertEqual(list(c.servo.previous_command.positions),goal)
        s=r.snapshot();s.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
        capture_snapshot(c,r,entry,s);self.assertIsNone(c.failure)
        self.assertEqual(c.last_snapshot['tick_contract']['row']['command_velocity'],[0.]*5)

    def test_world_clock_rewind_at_a_pause_is_terminal(self):
        c,r,_,_,_=self.setup_restore();r.world.current_time=.9
        with self.assertRaisesRegex(TickContractError,'World clock'):c.timeline(False)
        self.assertTrue(c.failure);self.assertTrue(c.generation.stalled)

    def test_missing_or_wrong_restore_identity_aborts_without_accepting_next_tick(self):
        for kind in ('missing','epoch','targets','world'):
            c,r,_,initial,_=self.setup_restore()
            def restore(tool,args):
                result=r.restore(initial,arm=True)
                if kind=='missing':result.pop('command_baseline')
                elif kind=='epoch':result['command_baseline']['epoch']=999
                elif kind=='targets':result['command_baseline']['drive_targets'][0]+=.01
                else:result['command_baseline']['physics_time_s']=-1.
                return result
            c.mutate_kit=restore
            with self.subTest(kind=kind),self.assertRaises((KeyError,TickContractError)):c.restore_physical('restore')
            self.assertTrue(c.failure);self.assertTrue(c.generation.stalled)


class IndependentClockTests(unittest.TestCase):
    def test_all_26_retained_clock_pairs_are_preserved_and_accepted(self):
        for pair in FIXTURE['clock_pairs']:
            s=checked_window([dict(sim_ns=pair['sim_ns'],physics_time_s=pair['physics_time_s'])])[0]
            self.assertEqual(s['physics_time_s'],pair['physics_time_s'])
            self.assertFalse(observation_errors(s['tick_contract']['row']))
        self.assertGreater(abs(FIXTURE['clock_pairs'][-1]['physics_time_s']*1e9-FIXTURE['clock_pairs'][-1]['sim_ns']),18000.)

    def test_actual_trial2_snapshot_passes_capture_without_clock_normalization(self):
        c,r,entry=probes.ReviewRegressions().collector();s=copy.deepcopy(FIXTURE['trial2'])
        c.generation.install(dict(epoch=s['epoch'],invalid_through_ns=s['sim_ns']-STEP_NS))
        r.epoch=s['epoch'];r.invalid_through_ns=s['sim_ns']-STEP_NS
        c.servo.plan=c.servo.grip_plan=None;c.servo.reference=list(s['drive_targets']);c.servo.reference_epoch=s['epoch']
        r.art.get_applied_actions().joint_positions[:]=[s['drive_targets']]
        capture_snapshot(c,r,entry,s)
        self.assertIsNone(c.failure)
        self.assertEqual(c.last_snapshot['physics_time_s'],s['physics_time_s'])
        self.assertGreater(abs(s['physics_time_s']*1e9-s['sim_ns']),2000.)

    def test_accumulated_drift_and_independent_origin_keep_strict_relative_steps(self):
        for base in (358.4833520296961,100000.+.005215):
            raw=[dict(sim_ns=358483333333+round(i*1e9/30),physics_time_s=base+i*4*FLOAT32_DT) for i in range(120)]
            self.assertTrue(require_snapshots(checked_window(raw))['passed'])
        for delta in (0.,-4*FLOAT32_DT,8*FLOAT32_DT,3*FLOAT32_DT):
            with self.subTest(delta=delta):self.assertTrue(world_step_errors(10.+delta,10.))

    def test_native_gap_is_rejected_even_when_world_advances_normally(self):
        rows=checked_window([dict(sim_ns=10**9,physics_time_s=1.),dict(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)])
        row=copy.deepcopy(rows[-1]['tick_contract']['row']);row['sim_ns']+=STEP_NS
        self.assertIn('Repeated or skipped physical tick',observation_errors(row,rows[0]['tick_contract']['row']))


class BoundaryChainTests(unittest.TestCase):
    def test_pause_then_restore_rejects_world_and_native_rewinds(self):
        for kind in ('world','native'):
            c,r,_,initial,_=RestoreHistoryTests().setup_restore()
            c.timeline(False)
            if kind=='world':r.world.current_time=.9
            else:r.stamp=lambda:900000000
            with self.assertRaises(Exception):c.restore_physical('restore')
            self.assertTrue(c.failure)

    def test_restore_then_same_receipt_cannot_replay_epoch(self):
        c,r,_,_,_=RestoreHistoryTests().setup_restore();receipt=c.restore_physical('restore')
        with self.assertRaises(Exception):c.cancel_owner(lambda:copy.deepcopy(receipt))
        self.assertTrue(c.failure)

    def test_consecutive_held_boundaries_preserve_clock_anchor_and_allow_next_tick(self):
        c,r,entry,_,_=RestoreHistoryTests().setup_restore()
        c.timeline(False);c.restore_physical('restore');c.restore_physical('restore')
        s=r.snapshot();s.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
        capture_snapshot(c,r,entry,s);self.assertIsNone(c.failure)

    def test_unobserved_multi_frame_gap_at_boundary_is_terminal(self):
        c,r,_,_,_=RestoreHistoryTests().setup_restore();c.timeline(False)
        r.stamp=lambda:1100000000;r.world.current_time=1.+12*FLOAT32_DT
        with self.assertRaises(TickContractError):c.restore_physical('restore')
        self.assertTrue(c.failure)
