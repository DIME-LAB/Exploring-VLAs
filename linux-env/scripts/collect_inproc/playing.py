"""Playing-timeline adapter: one app update owns physics and both RGB producers.

The setup follows w_31f/bench.py. Native runtime behavior is not yet verified in
this adapter; producer identity/cadence errors refuse the frame rather than poll.
"""
import copy
import time
from collections import Counter
import numpy as np
import shared
from native import Native
from tick_contract import STEP_NS,world_step_errors
from comparisons import matches


class FrameLedger:
    """Bind one fresh producer pair to one actual four-substep observation."""
    def __init__(self):self.previous=None;self.boundary=None

    def reset(self,epoch,stamp,world):
        self.previous=None;self.boundary=dict(epoch=epoch,sim_ns=stamp,physics_time_s=world)

    def accept(self,packet):
        previous=self.previous or self.boundary
        if previous is None:raise RuntimeError('Playing frame has no restored baseline')
        if packet['epoch']!=previous['epoch']:raise RuntimeError('Playing frame crosses reset epochs')
        if not matches(packet['sim_ns']-previous['sim_ns'],STEP_NS,'native_ns'):
            raise RuntimeError('Playing frame repeats, skips or rewinds physics time')
        if packet['native_physics_steps']!=4 or packet['world_step_count']!=4:
            raise RuntimeError('App update did not execute exactly four physical substeps')
        errors=world_step_errors(packet['physics_time_s'],previous['physics_time_s'])
        if errors:raise RuntimeError('Playing frame World clock: '+repr(errors))
        if not matches(packet['timeline_ns'],packet['sim_ns'],'native_ns'):
            raise RuntimeError('Playing timeline and physical clocks diverged')
        cameras=packet['cameras']
        if set(cameras)!={'wrist','top'}:raise RuntimeError('Playing frame lacks a camera')
        frame_ids=[]
        for key,camera in cameras.items():
            frame_id=camera['swh_frame_number']
            if not isinstance(frame_id,int) or frame_id<0:raise RuntimeError('Camera native frame identity is unavailable')
            if camera['completions']!=1:raise RuntimeError('Expected exactly one new render for '+key)
            if not matches(camera['stamp_ns'],packet['timeline_ns'],'stamp_ns'):
                raise RuntimeError('Rendered camera clock is not the sampled timeline: '+key)
            old=previous.get('cameras',{}).get(key)
            if old and (frame_id<=old['swh_frame_number'] or camera['copy_compute_count']<=old['copy_compute_count']):
                raise RuntimeError('Camera repeated an old render or annotation buffer: '+key)
            if camera['copy_compute_count']<=0:raise RuntimeError('RGB producer has not computed: '+key)
            frame_ids.append(frame_id)
        if len(set(frame_ids))!=1:raise RuntimeError('Camera pair belongs to different native render frames')
        self.previous=copy.deepcopy(packet)


class PlayingRGB:
    def __init__(self,paths):
        import omni.replicator.core as rep
        import omni.usd
        import carb.eventdispatcher
        import omni.syntheticdata._syntheticdata as sd
        self.products={};self.annotators={};self.references={}
        self.render_events=Counter();self.event_samples=[];self.calls=0
        interface=sd.acquire_syntheticdata_interface()
        def rendered(event):
            payload=interface.parse_rendered_simulation_event(event['product_path_handle'],event['results'])
            self.render_events[str(payload[0])]+=1
        self.observer=carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME,True),
            on_event=rendered,observer_name='inproc_playing_render_count')
        rep.orchestrator.set_capture_on_play(True)
        for key,path in paths.items():
            product=rep.create.render_product(path,(640,480))
            annotation=rep.AnnotatorRegistry.get_annotator('rgb',device='cuda')
            annotation.attach(product)
            reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            reference.attach(product)
            self.products[key]=product;self.annotators[key]=annotation;self.references[key]=reference

    def read(self,counts_before):
        images={};cameras={}
        start=time.perf_counter_ns()
        for key,product in self.products.items():
            data=self.annotators[key].get_data()
            data=data.numpy() if hasattr(data,'numpy') else np.asarray(data)
            if data.dtype!=np.uint8 or data.shape!=(480,640,4):raise RuntimeError('Playing RGB producer has no complete RGBA frame: '+key)
            info=product.hydra_texture.get_frame_info()
            reference=self.references[key].get_data()
            n,d=reference['referenceTimeNumerator'],reference['referenceTimeDenominator']
            if d<=0:raise RuntimeError('Camera reference clock has no denominator')
            cameras[key]=dict(swh_frame_number=info.get('swh_frame_number'),
                stamp_ns=round(int(n)/int(d)*1e9),reference_time=[int(n),int(d)],
                copy_compute_count=self.annotators[key].get_node().get_compute_count(),
                completions=self.render_events[product.path]-counts_before.get(product.path,0))
            images[key]=np.ascontiguousarray(data[:,:,:3])
        self.calls=min(camera['completions'] for camera in cameras.values())
        return images,cameras,(time.perf_counter_ns()-start)/1e6

    def close(self):
        for key,product in self.products.items():
            product.hydra_texture.set_updates_enabled(False)
            self.annotators[key].detach(product);self.references[key].detach(product)
            product.destroy()
        self.observer=None


class Playing(Native):
    def __init__(self,scene,manifest,warmup=120):
        import omni.timeline
        self.timeline=omni.timeline.get_timeline_interface()
        self.pending=None;self.frames=FrameLedger()
        super().__init__(scene,manifest,cameras=True)
        import omni.physx
        import carb
        from isaacsim.core.rendering_manager import RenderingManager
        settings=carb.settings.get_settings()
        settings.set('/app/asyncRendering',False)
        RenderingManager.set_dt(1/30)
        self.timeline.set_target_framerate(30.)
        self.timeline.set_auto_update(True);self.timeline.play()
        self.actual_physics_steps=0
        def physics_step(dt):self.actual_physics_steps+=1
        self.physics_subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step)
        for _ in range(warmup):
            self.align_timeline();self.world.step(render=True)
        self.restore(self.manifest['baseline'])

    def create_cameras(self):
        import omni.kit.viewport.utility as vu
        from pxr import UsdGeom
        viewport=vu.get_active_viewport()
        if viewport is None:
            self.engine_window=vu.create_viewport_window('inproc-playing-engine',width=640,height=480,visible=False)
            viewport=self.engine_window.viewport_api
        viewport.updates_enabled=False
        for path in self.manifest['camera_paths'].values():
            camera=self.stage.GetPrimAtPath(path)
            if not camera.IsA(UsdGeom.Camera):raise RuntimeError('Playing camera prim is missing')
            camera.ApplyAPI('OmniSensorAPI');camera.GetAttribute('omni:sensor:tickRate').Set(30.)
        self.rgb=PlayingRGB(self.manifest['camera_paths'])

    def restore(self,baseline):
        result=super().restore(baseline)
        self.align_timeline()
        self.pending=None
        self.frames.reset(result['epoch'],result['invalid_through_ns'],result['snapshot']['physics_time_s'])
        return result

    def align_timeline(self):
        # World play/pause pumps updates without physics, and float32 PhysX dt
        # accumulates differently from the timeline's 1/30. Own their alignment
        # without another app update; never rewrite the camera's original stamp.
        self.timeline.set_current_time(float(self.world.current_time))
        self.timeline.commit_silently()

    def step(self):
        if self.pending is not None:raise RuntimeError('Previous playing frame has not been consumed')
        if not self.world.is_playing():raise RuntimeError('Playing collector timeline unexpectedly stopped')
        self.align_timeline()
        start=time.perf_counter_ns();old_steps=self.actual_physics_steps
        index=self.world.current_time_step_index;before=dict(self.rgb.render_events)
        self.world.step(render=True)
        update_end=time.perf_counter_ns()
        timeline_ns=round(self.timeline.get_current_time()*1e9)
        snapshot=self.snapshot()
        images,cameras,copy_ms=self.rgb.read(before)
        packet=dict(epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],timeline_ns=timeline_ns,physics_time_s=snapshot['physics_time_s'],
            native_physics_steps=self.actual_physics_steps-old_steps,
            world_step_count=self.world.current_time_step_index-index,cameras=cameras)
        self.frames.accept(packet)
        profile=dict(app_update_ms=(update_end-start)/1e6,camera_read_ms=copy_ms,
            camera_stamps={k:v['stamp_ns'] for k,v in cameras.items()},playing_frame=packet,
            synchronization='one app update; native physics count; per-product SWH frame/copy counters')
        self.pending=(images,profile)
        return (time.perf_counter_ns()-start)/1e6

    def render(self):
        if self.pending is None:raise RuntimeError('No new playing frame is ready')
        packet=self.pending[1]['playing_frame']
        if packet['epoch']!=self.epoch or packet['sim_ns']!=self.stamp():raise RuntimeError('Playing RGB cache is stale')
        result=self.pending;self.pending=None
        return result

    def close(self):
        self.physics_subscription=None
        super().close()
