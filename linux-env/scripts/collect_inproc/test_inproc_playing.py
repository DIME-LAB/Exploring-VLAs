import copy
import unittest
from playing import FrameLedger


class PlayingTests(unittest.TestCase):
    def ledger(self):
        value=FrameLedger();value.reset(1,1000000000,1.)
        return value
    def packet(self,index=1):
        stamp=round((1+index/30)*1e9)
        return dict(epoch=1,sim_ns=stamp,timeline_ns=stamp,physics_time_s=1+index/30,native_physics_steps=4,world_step_count=4,
            cameras={key:dict(swh_frame_number=index,copy_compute_count=index,stamp_ns=stamp,completions=1)
                for key in ('wrist','top')})
    def test_exact_app_frame_and_pair(self):
        ledger=self.ledger();ledger.accept(self.packet());ledger.accept(self.packet(2))
    def test_camera_clock_is_retained_and_agreement_bound_applies(self):
        p=self.packet();p['cameras']['wrist']['stamp_ns']+=167
        ledger=self.ledger();ledger.accept(p)
        self.assertEqual(ledger.previous['cameras']['wrist']['stamp_ns'],p['sim_ns']+167)
    def test_missing_duplicate_lagged_or_extra_camera_fails(self):
        for mutation in ('missing','duplicate','lag','extra','different_frame','no_identity'):
            with self.subTest(mutation=mutation):
                p=self.packet()
                if mutation=='missing':del p['cameras']['wrist']
                if mutation=='duplicate':p['cameras']['wrist']['completions']=0
                if mutation=='extra':p['cameras']['wrist']['completions']=2
                if mutation=='lag':p['cameras']['wrist']['stamp_ns']-=33333333
                if mutation=='different_frame':p['cameras']['wrist']['swh_frame_number']=7
                if mutation=='no_identity':p['cameras']['wrist']['swh_frame_number']=None
                with self.assertRaises(RuntimeError):self.ledger().accept(p)
    def test_non_four_step_and_native_clock_faults(self):
        for key,value in (('native_physics_steps',3),('world_step_count',5),('epoch',0),('sim_ns',1000000000),('physics_time_s',1.)):
            with self.subTest(key=key):
                p=self.packet();p[key]=value
                with self.assertRaises(RuntimeError):self.ledger().accept(p)
    def test_old_rgb_buffer_and_old_frame_fail_even_when_stamp_is_new(self):
        ledger=self.ledger();ledger.accept(self.packet())
        p=self.packet(2);p['cameras']['top']['copy_compute_count']=1
        with self.assertRaises(RuntimeError):ledger.accept(p)
        p=self.packet(2)
        for v in p['cameras'].values():v['swh_frame_number']=1
        with self.assertRaises(RuntimeError):ledger.accept(p)
    def test_reset_invalidates_prior_epoch(self):
        ledger=self.ledger();ledger.accept(self.packet())
        ledger.reset(2,self.packet()['sim_ns'],self.packet()['physics_time_s'])
        with self.assertRaises(RuntimeError):ledger.accept(self.packet(2))

    def test_camera_lag_cannot_cancel_a_timeline_offset(self):
        p=self.packet();p['timeline_ns']+=33333333
        with self.assertRaises(RuntimeError):self.ledger().accept(p)

    def test_float32_world_clock_needs_per_update_alignment(self):
        import numpy as np
        ledger=self.ledger()
        for index in range(1,2001):
            p=self.packet(index)
            physical=1+index*4*float(np.float32(1/120))
            p['physics_time_s']=physical;p['sim_ns']=round(physical*1e9)
            p['timeline_ns']=round((1+(index-1)*4*float(np.float32(1/120))+1/30)*1e9)
            for camera in p['cameras'].values():camera['stamp_ns']=p['timeline_ns']
            ledger.accept(p)
        p=self.packet(2001);p['physics_time_s']=1+2001*4*float(np.float32(1/120))
        p['sim_ns']=round(p['physics_time_s']*1e9)
        with self.assertRaises(RuntimeError):ledger.accept(p)


if __name__=='__main__':unittest.main()
