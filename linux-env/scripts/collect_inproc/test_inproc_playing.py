import copy
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
from playing import FrameLedger,Playing,capture_settings,CAPTURE_KEYS,flush_physics_to_fabric,camera_parameters,render_pose_errors
from native import Native


class PlayingTests(unittest.TestCase):
    def test_post_render_camera_metadata_keeps_its_source_and_compute_count(self):
        import numpy as np
        node=Mock();node.get_type_name.return_value='omni.replicator.core.CameraParams'
        node.get_compute_count.return_value=7
        raw=dict(cameraViewTransform=np.eye(4),renderProductResolution=np.array([640,480]))
        annotation=Mock();annotation.get_data.return_value=raw;annotation.get_node.return_value=node
        result=camera_parameters(annotation)
        self.assertEqual(result['compute_count'],7)
        self.assertEqual(result['render_view'],np.eye(4).reshape(-1).tolist())
        node.get_type_name.return_value='other.CameraParams'
        with self.assertRaises(RuntimeError):camera_parameters(annotation)
        node.get_type_name.return_value='omni.replicator.core.CameraParams'
        raw['cameraViewTransform'][0,0]=float('nan')
        with self.assertRaises(RuntimeError):camera_parameters(annotation)

    def test_fabric_publication_uses_physical_clock_without_stepping_or_rendering(self):
        interface=Mock()
        world=SimpleNamespace(_physx_fabric_interface=interface,current_time=1.25,
            current_time_step_index=150,get_physics_dt=lambda:1/120)
        flush_physics_to_fabric(world)
        interface.force_update.assert_called_once_with(1/120,1.25)
        self.assertEqual(world.current_time_step_index,150)
        interface.force_update.side_effect=lambda *args:setattr(world,'current_time_step_index',154)
        with self.assertRaisesRegex(RuntimeError,'advanced physical time'):flush_physics_to_fabric(world)
        world._physx_fabric_interface=None;world._extension_manager=Mock()
        world._extension_manager.is_extension_enabled.return_value=False
        with self.assertRaisesRegex(RuntimeError,'requires enabled'):flush_physics_to_fabric(world)

    def test_restore_commits_play_only_when_resuming_a_paused_timeline(self):
        class Timeline:
            def __init__(self,playing):self.playing=playing;self.play_calls=0;self.commits=0
            def is_playing(self):return self.playing
            def set_auto_update(self,value):self.auto=value
            def play(self):self.play_calls+=1
            def commit(self):self.commits+=1;self.playing=True
            def set_current_time(self,value):self.time=value
            def get_current_time(self):return getattr(self,'time',1.)
            def commit_silently(self):pass
        class Settings:
            def __init__(self):self.values={}
            def set_bool(self,key,value):self.values[key]=value
            def get(self,key):return self.values.get(key)
        result=dict(epoch=7,invalid_through_ns=1000000000,snapshot=dict(physics_time_s=1.))
        for playing in (True,False):
            owner=Playing.__new__(Playing);owner.timeline=Timeline(playing)
            owner.prime_on_restore=False
            owner.settings=Settings();owner.frames=FrameLedger()
            owner.fabric_time=SimpleNamespace(Get=lambda:1.)
            texture=SimpleNamespace(set_updates_enabled=lambda enabled:None)
            owner.rgb=SimpleNamespace(products={'wrist':SimpleNamespace(hydra_texture=texture)})
            with patch.object(Native,'restore',return_value=result) as restore:
                owner.restore('baseline')
                restore.assert_called_once_with('baseline',manage_timeline=False)
            self.assertEqual(owner.timeline.play_calls,0 if playing else 1)
            self.assertEqual(owner.timeline.commits,0 if playing else 1)
            self.assertTrue(owner.timeline.is_playing());self.assertTrue(owner.timeline.auto)

    def test_aligned_native_clock_does_not_scrub_the_timeline(self):
        owner=Playing.__new__(Playing);owner.fabric_time=SimpleNamespace(Get=lambda:1.)
        owner.timeline=Mock();owner.timeline.get_current_time.return_value=1.
        owner.align_timeline()
        owner.timeline.set_current_time.assert_not_called()
        owner.timeline.commit_silently.assert_not_called()
        owner.timeline.get_current_time.return_value=.5
        owner.align_timeline()
        owner.timeline.set_current_time.assert_called_once_with(1.)
        owner.timeline.commit_silently.assert_called_once_with()

    def ledger(self):
        value=FrameLedger();value.reset(1,1000000000,1.)
        return value
    def packet(self,index=1):
        stamp=round((1+index/30)*1e9)
        observation=round((1+(index-1)/30)*1e9)
        return dict(epoch=1,sim_ns=stamp,observation_epoch=1,observation_ns=observation,timeline_ns=stamp,fabric_time_s=1+index/30,physics_time_s=1+index/30,native_physics_steps=4,world_step_count=4,
            cameras={key:dict(render_frame_number=index,copy_compute_count=index,stamp_ns=observation,completions=1,
                camera_params=dict(node_type='omni.replicator.core.CameraParams',compute_count=index,
                    source='SDG PostProcessRenderProductCamera renderResults',resolution=[640,480],
                    render_view=[float(i%5==0) for i in range(16)]))
                for key in ('wrist','top')})
    def test_exact_app_frame_and_pair(self):
        ledger=self.ledger();ledger.accept(self.packet());ledger.accept(self.packet(2))
    def test_synchronous_native_frame_can_be_buffered_for_its_own_observation(self):
        ledger=self.ledger()
        for index in (1,2):
            packet=self.packet(index)
            packet.update(image_delay_updates=0,observation_ns=packet['sim_ns'])
            for camera in packet['cameras'].values():camera['stamp_ns']=packet['sim_ns']
            ledger.accept(packet)
        packet=self.packet(3)
        packet['observation_ns']=ledger.previous['observation_ns']
        for camera in packet['cameras'].values():camera['stamp_ns']=packet['observation_ns']
        with self.assertRaises(RuntimeError):ledger.accept(packet)
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

    def test_historical_stamp_trace_is_not_a_render_pose_proof(self):
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
            previous=dict(epoch=3,sim_ns=row['before_ns'],physics_time_s=row['before_ns']/1e9)
            ledger.check_advance(packet(row),previous)
            ledger.reset(3,row['before_ns'],row['before_ns']/1e9)
            with self.assertRaisesRegex(RuntimeError,'Post-render camera metadata'):
                ledger.accept(packet(row))

    def test_missing_repeated_or_skipped_post_render_metadata_is_a_fault(self):
        packet=self.packet();del packet['cameras']['wrist']['camera_params']
        with self.assertRaises(RuntimeError):self.ledger().accept(packet)
        for count in (1,3):
            ledger=self.ledger();ledger.accept(self.packet())
            packet=self.packet(2);packet['cameras']['top']['camera_params']['compute_count']=count
            with self.assertRaises(RuntimeError):ledger.accept(packet)

    def test_measured_camera_params_trace_matches_pixels_while_drawable_metadata_lags(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/camera_params_32_trace.json').read_text())
        ledger=FrameLedger();epoch=None
        for frame in fixture['frames']:
            packet=frame['packet']
            if packet['epoch']!=epoch:
                epoch=packet['epoch'];ledger.reset(epoch,packet['observation_ns'],packet['physics_time_s']-1/30)
            ledger.accept(packet)
            render_pose_errors(packet['cameras'],frame['camera_world_before'])
        moving=fixture['frames'][16]
        corrupted=copy.deepcopy(moving['packet']['cameras'])
        corrupted['wrist']['camera_params']['render_view']=corrupted['wrist']['render_view']
        with self.assertRaisesRegex(RuntimeError,'Rendered camera pose differs'):
            render_pose_errors(corrupted,moving['camera_world_before'])

    def test_snapshot_refuses_invalid_handles_before_any_usd_fallback(self):
        for kind in ('articulation','body','link'):
            owner=Native.__new__(Native)
            views={name:SimpleNamespace(is_physics_handle_valid=lambda name=name:name!=kind)
                for name in ('articulation','body','link')}
            owner.art=views['articulation'];owner.bodies={'body':views['body']};owner.links={'link':views['link']}
            with self.assertRaisesRegex(RuntimeError,'Invalid measured physics handle: '+kind):owner.snapshot()


if __name__=='__main__':unittest.main()
