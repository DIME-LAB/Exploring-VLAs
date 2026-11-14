import copy
import unittest
import json
from pathlib import Path
from playing import FrameLedger,capture_settings,CAPTURE_KEYS


class PlayingTests(unittest.TestCase):
    def ledger(self):
        value=FrameLedger();value.reset(1,1000000000,1.)
        return value
    def packet(self,index=1):
        stamp=round((1+index/30)*1e9)
        observation=round((1+(index-1)/30)*1e9)
        return dict(epoch=1,sim_ns=stamp,observation_epoch=1,observation_ns=observation,timeline_ns=stamp,fabric_time_s=1+index/30,physics_time_s=1+index/30,native_physics_steps=4,world_step_count=4,
            cameras={key:dict(render_frame_number=index,copy_compute_count=index,stamp_ns=observation,completions=1)
                for key in ('wrist','top')})
    def test_exact_app_frame_and_pair(self):
        ledger=self.ledger();ledger.accept(self.packet());ledger.accept(self.packet(2))
    def test_camera_clock_is_retained_and_agreement_bound_applies(self):
        p=self.packet();p['cameras']['wrist']['stamp_ns']+=167
        ledger=self.ledger();ledger.accept(p)
        self.assertEqual(ledger.previous['cameras']['wrist']['stamp_ns'],p['observation_ns']+167)
    def test_missing_duplicate_lagged_or_extra_camera_fails(self):
        for mutation in ('missing','duplicate','lag','extra','different_frame','no_identity'):
            with self.subTest(mutation=mutation):
                p=self.packet()
                if mutation=='missing':del p['cameras']['wrist']
                if mutation=='duplicate':p['cameras']['wrist']['completions']=0
                if mutation=='extra':p['cameras']['wrist']['completions']=2
                if mutation=='lag':p['cameras']['wrist']['stamp_ns']-=33333333
                if mutation=='different_frame':p['cameras']['wrist']['render_frame_number']=7
                if mutation=='no_identity':p['cameras']['wrist']['render_frame_number']=None
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
        for v in p['cameras'].values():v['render_frame_number']=1
        with self.assertRaises(RuntimeError):ledger.accept(p)
    def test_reset_invalidates_prior_epoch(self):
        ledger=self.ledger();ledger.accept(self.packet())
        ledger.reset(2,self.packet()['sim_ns'],self.packet()['physics_time_s'])
        with self.assertRaises(RuntimeError):ledger.accept(self.packet(2))

    def test_camera_lag_cannot_cancel_a_timeline_offset(self):
        p=self.packet();p['timeline_ns']+=33333333
        with self.assertRaises(RuntimeError):self.ledger().accept(p)

    def test_native_graph_clock_with_float32_world_accumulation(self):
        import numpy as np
        ledger=self.ledger()
        for index in range(1,2001):
            p=self.packet(index)
            physical=1+index*4*float(np.float32(1/120))
            p['physics_time_s']=physical
            ledger.accept(p)
        p=self.packet(2001);p['physics_time_s']=1+2001*4*float(np.float32(1/120))
        p['timeline_ns']=round(p['physics_time_s']*1e9)
        with self.assertRaises(RuntimeError):ledger.accept(p)

    def test_both_capture_settings_survive_a_runtime_reset(self):
        class Settings:
            def __init__(self):self.values={key:False for key in CAPTURE_KEYS}
            def get(self,key):return self.values[key]
            def set_bool(self,key,value):self.values[key]=value
        settings=Settings()
        with self.assertRaises(RuntimeError):capture_settings(settings)
        capture_settings(settings,enable=True)
        # Installed Orchestrator.reset copies the persistent flag to runtime.
        settings.values[CAPTURE_KEYS[1]]=settings.values[CAPTURE_KEYS[0]]
        self.assertTrue(all(capture_settings(settings).values()))
        settings.values[CAPTURE_KEYS[1]]=False
        with self.assertRaises(RuntimeError):capture_settings(settings)

    def test_fabric_clock_must_match_the_native_step(self):
        p=self.packet();p['fabric_time_s']-=1/30
        with self.assertRaises(RuntimeError):self.ledger().accept(p)

    def test_join_rejects_wrong_observation_or_skipped_copy(self):
        for key,value in (('observation_ns',966666667),('observation_epoch',0)):
            p=self.packet();p[key]=value
            with self.assertRaises(RuntimeError):self.ledger().accept(p)
        ledger=self.ledger();ledger.accept(self.packet())
        p=self.packet(2);p['cameras']['wrist']['copy_compute_count']+=1
        with self.assertRaises(RuntimeError):ledger.accept(p)

    def test_measured_one_update_trace_preserves_original_stamps(self):
        trace=json.loads((Path(__file__).parent/'fixtures/playing_v11_trace.json').read_text())['trace']
        def packet(row):
            # The World clock here is a test model. The fixture retains the real
            # native graph/camera clocks and callback counts; float32 World drift
            # is exercised independently above.
            return dict(epoch=3,sim_ns=row['after_ns'],timeline_ns=row['after_ns'],
                fabric_time_s=row['after_ns']/1e9,physics_time_s=row['after_ns']/1e9,
                observation_epoch=3,observation_ns=row['before_ns'],
                native_physics_steps=row['physical_steps'],world_step_count=row['world_steps'],cameras=row['cameras'])
        ledger=FrameLedger();ledger.reset(3,trace[0]['before_ns'],trace[0]['before_ns']/1e9)
        for row in trace:
            ledger.accept(packet(row))
            self.assertEqual(ledger.previous['cameras'],row['cameras'])
        ledger=FrameLedger();ledger.reset(3,trace[0]['before_ns'],trace[0]['before_ns']/1e9)
        ledger.accept(packet(trace[0]))
        with self.assertRaises(RuntimeError):ledger.accept(packet(trace[2]))


if __name__=='__main__':unittest.main()
