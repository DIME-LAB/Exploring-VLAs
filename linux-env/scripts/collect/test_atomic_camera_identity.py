"""A reset cannot publish half an identity or split stale filtering from binding."""
import threading,unittest
from unittest.mock import patch
from protocol import StepGeneration
from test_sensor_join import sensors,publisher_info
from test_frame_join import image

class AtomicIdentityTests(unittest.TestCase):
    def test_camera_waits_for_complete_startup_or_reset_identity_then_drops_stale(self):
        for startup in (False,True):
            with self.subTest(startup=startup):
                s=sensors()
                if startup:s.generation=StepGeneration()
                entered=threading.Event();release=threading.Event();original=s.generation._identity_lock
                class Gate:
                    def __enter__(self):
                        original.acquire()
                        if threading.current_thread().name=='installer':entered.set();self.check=release.wait(2);assert self.check
                    def __exit__(self,*args):original.release()
                s.generation._identity_lock=Gate();errors=[]
                def install():
                    try:s.generation.install(dict(epoch=2,invalid_through_ns=4*10**9))
                    except BaseException as exc:errors.append(exc)
                worker=threading.Thread(target=install,name='installer');worker.start();self.assertTrue(entered.wait(2))
                callback=threading.Thread(target=lambda:s.put('wrist',image(3*10**9,1),publisher_info(8)));callback.start()
                self.assertFalse(s.camera_publishers);release.set();worker.join(2);callback.join(2)
                self.assertFalse(worker.is_alive());self.assertFalse(callback.is_alive());self.assertEqual(errors,[])
                self.assertEqual(s.generation.identity,(2,4*10**9));self.assertFalse(s.camera_publishers);self.assertIsNone(s.failure)
                self.assertEqual(s.stale_camera_messages['wrist'],1)

    def test_install_cannot_cross_the_camera_filter_and_bind_decision(self):
        s=sensors();entered=threading.Event();release=threading.Event();installed=threading.Event()
        scope=s.put.__globals__;original=scope['publisher_identity']
        def paused(info):entered.set();assert release.wait(2);return original(info)
        def install():s.generation.install(dict(epoch=2,invalid_through_ns=4*10**9));installed.set()
        with patch.dict(scope,publisher_identity=paused):
            callback=threading.Thread(target=lambda:s.put('wrist',image(3*10**9,1),publisher_info()));callback.start();self.assertTrue(entered.wait(2))
            worker=threading.Thread(target=install);worker.start();self.assertFalse(installed.wait(.02));release.set();callback.join(2);worker.join(2)
        self.assertTrue(installed.is_set());self.assertEqual(s.camera_first_frame['wrist']['receipt_epoch'],1)

    def test_durable_history_prunes_only_fenced_images_and_keeps_stream_binding(self):
        s=sensors();s.put('wrist',image(10**9,1),publisher_info());s.put('wrist',image(10**9,2),publisher_info())
        self.assertTrue(s.rerender_log)
        s.generation.install(dict(epoch=2,invalid_through_ns=10**9));s.prune_durable_camera_history()
        self.assertFalse(s.seen_camera_frames['wrist']);self.assertFalse(s.rerender_log);self.assertTrue(s.camera_publishers)
        s.put('wrist',image(10**9,3),publisher_info(9));self.assertIsNone(s.failure)
