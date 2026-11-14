import unittest
import numpy as np
from rgb_capture import payload_images


class RGBTests(unittest.TestCase):
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
