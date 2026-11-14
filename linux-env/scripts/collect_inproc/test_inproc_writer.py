import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from writer import Writer
from compare import image_metrics,stage_summary


@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required for real writer check')
class WriterTests(unittest.TestCase):
    def test_raw_rows_and_both_videos_are_complete(self):
        with tempfile.TemporaryDirectory() as directory,patch('writer.confined',lambda p:Path(p)):
            output=Path(directory)/'attempt'
            writer=Writer(output)
            images={k:np.full((480,640,3),[32,96,160],np.uint8) for k in ('wrist','top')}
            expected=[dict(sim_ns=(i+1)*33333333,index=i) for i in range(3)]
            for row in expected:writer.add(row,images,dict(index=row['index']),{})
            writer.close()
            self.assertEqual([json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()],expected)
            self.assertEqual(len((output/'poses.jsonl').read_text().splitlines()),3)
            for key in images:
                raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(output/(key+'.mp4')),
                    '-f','rawvideo','-pix_fmt','rgb24','-'])
                self.assertEqual(len(raw),3*480*640*3)
                frames=np.frombuffer(raw,np.uint8).reshape(3,480,640,3)
                self.assertLess(float(np.mean(abs(frames[0].astype(float)-images[key]))),3.)
    def test_bad_layout_refuses_before_encoder_queue(self):
        with tempfile.TemporaryDirectory() as directory,patch('writer.confined',lambda p:Path(p)):
            writer=Writer(Path(directory)/'attempt',render=False)
            with self.assertRaises(ValueError):writer.add(dict(sim_ns=1),{'wrist':np.zeros((10,10,3),np.uint8)},None,{})
            writer.close()
    def test_pixel_metrics_and_stage_totals(self):
        a=np.zeros((480,640,3),np.uint8)
        same=image_metrics(a,a);self.assertTrue(same['identical']);self.assertAlmostEqual(same['ssim'],1.)
        b=a.copy();b[:]=10
        self.assertFalse(image_metrics(a,b)['identical'])
        self.assertEqual(stage_summary([dict(physics_ms=1.),dict(physics_ms=3.)])['physics_ms']['total_ms'],4.)


if __name__=='__main__':unittest.main()
