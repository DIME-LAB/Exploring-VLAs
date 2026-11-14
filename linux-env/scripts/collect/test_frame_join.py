"""Competing pixels cannot enter a valid command/recording, in either arrival order."""
import collections,copy,json,threading,unittest
from types import SimpleNamespace as NS
import test_review_regressions as probes
from frame_join import SensorAmbiguity
from test_sensor_join import sensors,feed,message,publisher_info


def image(stamp,value):
    return NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)),encoding='rgb8',width=640,height=480,
              step=1920,is_bigendian=0,data=bytes([value])*921600)


class ConflictingFrameTests(unittest.TestCase):
    def test_conflicting_camera_arrivals_poison_actual_capture_without_command_or_row(self):
        native=10**9
        for stamps,values in [((native,native+500),(1,2)),((native+500,native),(2,1)),
                              ((native-500,native+500),(1,2)),((native,native),(1,2)),
                              ((native,native+167),(1,1))]:
            with self.subTest(stamps=stamps,values=values):
                c,r,entry=probes.ReviewRegressions().collector();s=sensors();s.generation=c.generation
                s.put('wrist',image(stamps[0],values[0]),publisher_info())
                with self.assertRaises(SensorAmbiguity):s.put('wrist',image(stamps[1],values[1]),publisher_info(3))
                c.sensors=s;rows=[];c.writer=NS(add=lambda *args:rows.append(args))
                c.capture()
                self.assertTrue(c.failure);self.assertTrue(c.generation.stalled);self.assertEqual(c.events,['pause'])
                self.assertEqual(rows,[])
                evidence=c.capture_failures[0]['sensor_join_failure']
                self.assertEqual(len(evidence['candidates']),2)
                self.assertEqual({p['stamp_ns'] for p in evidence['candidates']},set(stamps))
                self.assertTrue(all(p['content_sha256'] for p in evidence['candidates']))

    def test_exact_image_duplicate_is_allowed_but_a_late_conflict_is_terminal(self):
        native=1033333333;s=sensors();first=image(native+167,1)
        s.put('wrist',first,publisher_info());s.put('wrist',copy.deepcopy(first),publisher_info());self.assertEqual(s.duplicate_messages['wrist'],1)
        s.put('top',image(native+167,3),publisher_info(2));s.put('joint',message(native));s.put('tick',NS(data=json.dumps(dict(sim_ns=native,epoch=1))))
        self.assertEqual(s.read(timeout=0)[0],native+167)
        s.put('wrist',copy.deepcopy(first),publisher_info());self.assertEqual(s.duplicate_messages['wrist'],2)
        with self.assertRaises(SensorAmbiguity):s.put('wrist',image(native+167,2),publisher_info(3))
        with self.assertRaises(SensorAmbiguity):s.read(timeout=0)

    def test_distinct_joint_candidate_is_not_overwritten(self):
        native=10**9;s=sensors();a=message(native);b=message(native)
        a.position=[0.]*6;b.position=[.1]*6
        s.put('joint',a)
        with self.assertRaises(SensorAmbiguity):s.put('joint',b)
        self.assertEqual(s.pending[native]['joint'][0].position,[0.]*6)
