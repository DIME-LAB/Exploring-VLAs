import unittest
import numpy as np
from rgb_capture import payload_images,RGB
from unittest.mock import patch
from types import ModuleType,SimpleNamespace


class RGBTests(unittest.TestCase):
    def test_render_products_are_disabled_after_a_failed_capture(self):
        states=[]
        rgb=object.__new__(RGB)
        rgb.start_capture=lambda:SimpleNamespace(done=lambda:False,cancel=lambda:None)
        rgb.products={'wrist':SimpleNamespace(hydra_texture=SimpleNamespace(set_updates_enabled=states.append))}
        module=ModuleType('omni.replicator.core')
        def fail(**kwargs):raise RuntimeError('render failure')
        module.orchestrator=SimpleNamespace(step=fail)
        omni=ModuleType('omni');replicator=ModuleType('omni.replicator')
        omni.replicator=replicator;replicator.core=module
        with patch.dict('sys.modules',{'omni':omni,'omni.replicator':replicator,'omni.replicator.core':module}):
            with self.assertRaises(RuntimeError):rgb.read(1/30,fail)
        self.assertEqual(states,[True,False])
        self.assertIsNone(rgb.expected)

    def test_paired_payload_is_copied_and_time_checked(self):
        frame=np.zeros((480,640,4),np.uint8)
        data={'reference_time':(1,30),'rgb-A':frame,'rgb-B':frame}
        result=payload_images(data,dict(wrist='A',top='B'),1/30)
        frame[:]=255
        self.assertFalse(result['top'].any())
        with self.assertRaises(RuntimeError):payload_images(data,dict(wrist='A',top='B'),2/30)
        del data['rgb-B']
        with self.assertRaises(KeyError):payload_images(data,dict(wrist='A',top='B'),1/30)


if __name__=='__main__':unittest.main()
