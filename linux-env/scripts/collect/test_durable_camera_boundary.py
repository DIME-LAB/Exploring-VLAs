"""Provisional images cannot disappear across a certified episode boundary."""
import threading,unittest
from protocol import CancelledGeneration
from frame_join import SensorAmbiguity
from test_native_camera_tick import setup,native
from test_sensor_join import message,publisher_info
from test_frame_join import image


class DurableBoundaryTests(unittest.TestCase):
    def test_review_three_header_orphan_is_terminal_at_boundary(self):
        s,tick=setup();native(s,tick)
        s.put('wrist',image(tick-1000,11),publisher_info());s.put('wrist',image(tick,22),publisher_info())
        s.put('wrist',image(tick+1001,33),publisher_info())
        s.put('top',image(tick,44),publisher_info(2));s.put('joint',message(tick))
        self.assertEqual(s.read(timeout=0)[1]['wrist'][0].data[0],11)
        with self.assertRaisesRegex(SensorAmbiguity,'durable boundary') as error:s.require_camera_boundary(tick,timeout=0)
        self.assertEqual(error.exception.evidence['candidates'][0]['stamp_ns'],tick+1001)
        self.assertEqual(s.camera_boundary_floor,-1)

    def test_late_orphan_cannot_arrive_after_a_successful_boundary_check(self):
        s,tick=setup();native(s,tick)
        s.put('wrist',image(tick-1000,11),publisher_info());s.put('wrist',image(tick,22),publisher_info())
        self.assertTrue(s.require_camera_boundary(tick,timeout=0)['passed'])
        with self.assertRaisesRegex(SensorAmbiguity,'durable boundary'):s.put('wrist',image(tick+1001,33),publisher_info())

    def test_future_camera_first_remains_provisional_until_its_native_delivery(self):
        s,tick=setup();native(s,tick);s.put('wrist',image(tick-2,11),publisher_info())
        s.require_camera_boundary(tick,timeout=0)
        future=tick+33333333;s.put('wrist',image(future-2,22),publisher_info())
        self.assertIsNone(s.failure)
        native(s,future)
        self.assertTrue(s.require_camera_boundary(future,timeout=0)['passed'])

    def test_boundary_waits_for_inflight_native_publication_with_a_wall_deadline(self):
        s,tick=setup();s.put('wrist',image(tick-2,11),publisher_info())
        entered=threading.Event();original=s.condition.wait
        def wait(timeout):entered.set();return original(timeout)
        s.condition.wait=wait;result=[];errors=[]
        def boundary():
            try:result.append(s.require_camera_boundary(tick,timeout=1))
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=boundary);worker.start();self.assertTrue(entered.wait(1));native(s,tick);worker.join(2)
        self.assertFalse(worker.is_alive());self.assertFalse(errors);self.assertTrue(result[0]['passed'])
