import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from wrist_retention import WristHistory

def sample(i,grip=1.,epoch=9):
    stamp=round(i*1e9/30)
    snapshot=dict(sim_ns=stamp,epoch=epoch,state=[0.]*5+[grip])
    identity=dict(stream='wrist',stamp_ns=stamp,native_tick=dict(epoch=epoch,sim_ns=stamp))
    return snapshot,np.full((480,640,3),i%256,np.uint8),identity

class WristRetentionTests(unittest.TestCase):
    def test_disabled_never_reads_images_or_creates_output(self):
        history=WristHistory()
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'none';history.observe(None,None,None,None)
            self.assertEqual(history.finish(out),dict(enabled=False));self.assertFalse(out.exists())

    def test_one_second_boundary_is_lossless_and_input_is_not_mutated(self):
        h=WristHistory(True,max_actions=45)
        for i in range(46):
            s,image,identity=sample(i,1. if i<30 else .9);before=image.copy()
            h.observe(s,image,identity,i);self.assertTrue(np.array_equal(image,before));image.fill(255)
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'frames';summary=h.finish(out);m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(summary['frames_saved'],31);self.assertEqual(m['closures'][0]['frame_sim_ns'],[round(i*1e9/30) for i in range(31)])
            self.assertFalse(m['closures'][0]['episode_start_truncated'])
            for frame in m['frames']:
                with Image.open(out/frame['file']) as im:raw=np.asarray(im).tobytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),frame['raw_rgb_sha256'])
                self.assertEqual(raw[0],frame['executed_actions'])
            with self.assertRaises(ValueError):h.finish(Path(t)/'again')

    def test_multiple_closures_are_deduplicated_and_memory_is_bounded(self):
        h=WristHistory(True,max_actions=90)
        for i in range(91):
            grip=1. if i<30 or 60<=i<75 else .8
            s,image,identity=sample(i,grip);h.observe(s,image,identity,i)
        self.assertEqual(len(h.events),2);self.assertEqual(len(h.selected),62)
        self.assertLessEqual(h.peak_frames,91)

    def test_epoch_skip_stale_and_incomplete_refuse(self):
        for bad in ('epoch','skip','stale'):
            h=WristHistory(True,max_actions=2);h.observe(*sample(0),0)
            s,image,identity=sample(1)
            if bad=='epoch':s['epoch']=10
            if bad=='skip':s,image,identity=sample(2)
            if bad=='stale':identity['native_tick']['sim_ns']-=10000
            with self.subTest(bad=bad),self.assertRaises(ValueError):h.observe(s,image,identity,1)
        h=WristHistory(True,max_actions=2);h.observe(*sample(0),0)
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):h.finish(Path(t)/'partial')
        # A fresh per-episode object admits a new epoch/time origin.
        new=WristHistory(True,max_actions=1);new.observe(*sample(0,epoch=10),0)

if __name__=='__main__':unittest.main()
