"""Original producer stamps enter actual Sensors.put/read; no prejoined frame shortcut."""
import collections,json,threading,unittest
from types import SimpleNamespace as NS
import test_review_regressions as probes
from contract import JOINTS
from protocol import StepGeneration
from tick_contract import TickContract
import test_restore_clock as restore_probes
from test_restore_clock import capture_snapshot
from tick_contract import FLOAT32_DT


def sensors():
    Sensors,scope=probes.load_class('collect.py','Sensors');scope['collections']=collections
    s=object.__new__(Sensors);s.condition=threading.Condition();s.pending=collections.OrderedDict()
    s.last=-1;s.failure=None;s.duplicate_messages=collections.Counter();s.generation=StepGeneration()
    return s


def message(stamp):return NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)))


def feed(s,native,offset):
    for key in ('wrist','joint','top','tick'):
        s.put(key,NS(data=json.dumps(dict(sim_ns=native,epoch=1))) if key=='tick' else message(native+(offset if key in ('wrist','top') else 0)))


class SensorJoinTests(unittest.TestCase):
    def test_adjacent_buckets_and_each_edge_of_producer_tolerance_join_unchanged(self):
        native=1033333333
        for offset in (-1000,-500,-167,-11,0,11,166,167,500,1000):
            with self.subTest(offset=offset):
                s=sensors();feed(s,native,offset);stamp,entry=s.read(timeout=0)
                self.assertEqual(stamp,native+offset);self.assertEqual(entry['tick'][0]['sim_ns'],native)
                self.assertEqual(entry['joint'][0].header.stamp.nanosec,native%10**9)
                s.put('wrist',message(native+offset));self.assertEqual(s.duplicate_messages['wrist'],1)
                with self.assertRaises(TimeoutError):s.read(timeout=0)
                feed(s,native+33333333,offset);self.assertEqual(s.read(timeout=0)[0],native+33333333+offset)

    def test_outside_tolerance_or_missing_frame_never_joins(self):
        for offset in (-1001,1001,33333333):
            with self.subTest(offset=offset):
                s=sensors();feed(s,1033333333,offset)
                with self.assertRaises(TimeoutError):s.read(timeout=0)

    def test_ambiguous_nearby_candidates_fail_instead_of_arbitrarily_joining(self):
        s=sensors();stamp=1033333333
        s.put('joint',message(stamp));s.put('top',message(stamp+1500))
        s.put('wrist',message(stamp+750))
        with self.assertRaisesRegex(RuntimeError,'Ambiguous'):s.read(timeout=0)


class NativeAnchorTests(unittest.TestCase):
    def test_actual_restore_capture_and_held_cancel_accept_the_full_producer_tolerance(self):
        for offset in (-1000,-500,-11,0,11,500,1000):
            with self.subTest(offset=offset):
                c,r,entry,_,_=restore_probes.RestoreHistoryTests().setup_restore();c.restore_physical('restore')
                snapshot=r.snapshot();snapshot.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
                capture_snapshot(c,r,entry,snapshot,camera_offset_ns=offset)
                self.assertIsNone(c.failure)
                c.timeline(False);self.assertIsNone(c.failure)

    def test_native_anchor_stays_strict_when_camera_skew_is_legal(self):
        from tick_contract import TickContractError
        for bad in (-1,1):
            c,r,entry,_,_=restore_probes.RestoreHistoryTests().setup_restore();c.restore_physical('restore')
            snapshot=r.snapshot();snapshot.update(sim_ns=1033333333,physics_time_s=1.+4/120)
            capture_snapshot(c,r,entry,snapshot,camera_offset_ns=500);self.assertIsNone(c.failure)
            r.stamp=lambda:1033333333+bad*33333333
            r.world.current_time=1.+(4+bad*4)/120
            if bad<0:
                with self.assertRaises(Exception):c.timeline(False)
                self.assertTrue(c.failure)
            else:
                # One actually completed in-flight frame is legal; skipping two is not.
                r.stamp=lambda:1033333333+2*33333333;r.world.current_time=1.+12/120
                with self.assertRaises(TickContractError):c.timeline(False)
                self.assertTrue(c.failure)

    def test_actual_restore_capture_refuses_outside_producer_tolerance(self):
        for offset in (-1001,1001,33333333):
            c,r,entry,_,_=restore_probes.RestoreHistoryTests().setup_restore();c.restore_physical('restore')
            snapshot=r.snapshot();snapshot.update(sim_ns=1033333333,physics_time_s=1.+4*FLOAT32_DT)
            before=len(c.events);capture_snapshot(c,r,entry,snapshot,camera_offset_ns=offset)
            self.assertTrue(c.failure);self.assertEqual(c.events[before:],['pause'])

    def test_native_cadence_accepts_independent_camera_rounding_that_changes_between_ticks(self):
        import copy
        from tick_contract import require_snapshots
        from contract import timing_audit
        c,r,entry,_,_=restore_probes.RestoreHistoryTests().setup_restore();c.restore_physical('restore')
        held=[]
        for index,offset in enumerate((1000,-1000,167),1):
            snapshot=r.snapshot();snapshot.update(sim_ns=10**9+round(index*1e9/30),physics_time_s=1.+index*4*FLOAT32_DT)
            capture_snapshot(c,r,entry,snapshot,camera_offset_ns=offset)
            self.assertIsNone(c.failure);held.append(copy.deepcopy(c.last_snapshot))
        self.assertTrue(require_snapshots(held)['passed'])
        audit=timing_audit([dict(s['tick_contract']['row'],executing=True) for s in held],30)
        self.assertTrue(audit['contract_passed']);self.assertAlmostEqual(audit['sim_hz'],30,places=5)
