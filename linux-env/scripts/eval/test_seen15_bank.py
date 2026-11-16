import copy,unittest
from prepare_seen15_bank import check_frame,SELECTION
class Seen15Tests(unittest.TestCase):
    def test_verified_recording_selection(self):
        seeds=[x[2] for x in SELECTION]
        self.assertEqual(seeds,list(range(530000,530006))+list(range(530500,530506))+list(range(531000,531003)))
        self.assertNotIn(530006,seeds)
    def test_one_tick_state_and_camera_join(self):
        start=dict(sim_ns=1000000000,state=[0]*6);frame=dict(sim_ns=1033333333,state=[0]*6,observation_stamp_ns=1033333333,top_stamp_ns=1033333333,wrist_stamp_ns=1033333333)
        self.assertEqual(check_frame(frame,start),0)
        for key in ('top_stamp_ns','wrist_stamp_ns','observation_stamp_ns'):
            bad=copy.deepcopy(frame);bad[key]+=1
            with self.assertRaises(ValueError):check_frame(bad,start)
        bad=copy.deepcopy(frame);bad['state'][0]=.001
        with self.assertRaises(ValueError):check_frame(bad,start)
        bad=copy.deepcopy(frame);bad['sim_ns']+=33333333
        for key in ('top_stamp_ns','wrist_stamp_ns','observation_stamp_ns'):bad[key]=bad['sim_ns']
        with self.assertRaises(ValueError):check_frame(bad,start)
if __name__=='__main__':unittest.main()
