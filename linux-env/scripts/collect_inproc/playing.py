"""Playing-timeline adapter: one app update owns physics and both RGB producers.

The setup follows w_31f/bench.py. Native runtime behavior is not yet verified in
this adapter; producer identity/cadence errors refuse the frame rather than poll.
"""
import copy
import math
import time
from collections import Counter
import numpy as np
import shared
from native import Native
from tick_contract import STEP_NS,world_step_errors
from comparisons import matches

CAPTURE_KEYS=('/persistent/omni/replicator/captureOnPlay','/omni/replicator/captureOnPlay')


def flush_physics_to_fabric(world):
    """Publish the measured pre-update pose without a render or physical step."""
    if world._physx_fabric_interface is None:
        if not world._extension_manager.is_extension_enabled('omni.physx.fabric'):
            raise RuntimeError('Playing capture requires enabled omni.physx.fabric')
        from omni.physxfabric import get_physx_fabric_interface
        world._physx_fabric_interface=get_physx_fabric_interface()
    before=(world.current_time,world.current_time_step_index)
    started=time.perf_counter_ns()
    # Same publication as installed SimulationContext.render(), without its
    # extra held app update. World.step(render=True) omits this explicit flush.
    world._physx_fabric_interface.force_update(world.get_physics_dt(),world.current_time)
    if (world.current_time,world.current_time_step_index)!=before:
        raise RuntimeError('Fabric publication advanced physical time')
    return (time.perf_counter_ns()-started)/1e6


def capture_settings(settings,enable=False):
    if enable:
        for key in CAPTURE_KEYS:settings.set_bool(key,True)
    values={key:settings.get(key) for key in CAPTURE_KEYS}
    if not all(value is True for value in values.values()):
        raise RuntimeError('Replicator capture-on-play is not enabled: '+repr(values))
    return values


class FrameLedger:
    """Bind one fresh producer pair to one actual four-substep observation."""
    def __init__(self):self.previous=None;self.boundary=None

    def reset(self,epoch,stamp,world,cameras=None):
        self.previous=None;self.boundary=dict(epoch=epoch,sim_ns=stamp,physics_time_s=world)
        if cameras is not None:self.boundary['cameras']=copy.deepcopy(cameras)

    def check_advance(self,packet,previous):
        if previous is None:raise RuntimeError('Playing frame has no restored baseline')
        if packet['epoch']!=previous['epoch']:raise RuntimeError('Playing frame crosses reset epochs')
        if not matches(packet['sim_ns']-previous['sim_ns'],STEP_NS,'native_ns'):
            raise RuntimeError('Playing frame repeats, skips or rewinds physics time')
        if packet['native_physics_steps']!=4 or packet['world_step_count']!=4:
            raise RuntimeError('App update did not execute exactly four physical substeps')
        errors=world_step_errors(packet['physics_time_s'],previous['physics_time_s'])
        if errors:raise RuntimeError('Playing frame World clock: '+repr(errors))
        if not matches(packet['timeline_ns'],packet['sim_ns'],'native_ns'):
            raise RuntimeError('Playing timeline and native graph clocks diverged')
        if not matches(round(packet['fabric_time_s']*1e9),packet['sim_ns'],'native_ns'):
            raise RuntimeError('Fabric renderer time and native graph clocks diverged')

    def accept(self,packet):
        previous=self.previous or self.boundary
        self.check_advance(packet,previous)
        delay=packet.get('image_delay_updates',1)
        if delay not in (0,1):raise RuntimeError('Image delivery exceeds the one-update bound')
        expected=packet['sim_ns'] if delay==0 else previous['sim_ns']
        if (packet['observation_epoch']!=packet['epoch'] or not matches(
                packet['observation_ns'],expected,'native_ns')):
            raise RuntimeError('Image join lacks its measured physical observation')
        if self.previous and not matches(packet['observation_ns']-self.previous['observation_ns'],STEP_NS,'native_ns'):
            raise RuntimeError('Image observation stamp repeated or skipped')
        cameras=packet['cameras']
        if set(cameras)!={'wrist','top'}:raise RuntimeError('Playing frame lacks a camera')
        frame_ids=[]
        for key,camera in cameras.items():
            frame_id=camera['render_frame_number']
            if not isinstance(frame_id,int) or frame_id<0:raise RuntimeError('Camera native frame identity is unavailable')
            if camera['completions']!=1:raise RuntimeError('Expected exactly one new render for '+key)
            if not matches(camera['stamp_ns'],packet['observation_ns'],'stamp_ns'):
                raise RuntimeError('Camera frame does not match the retained observation: '+key)
            old=previous.get('cameras',{}).get(key)
            if old and (frame_id<=old['render_frame_number'] or camera['copy_compute_count']!=old['copy_compute_count']+1):
                raise RuntimeError('Camera repeated or skipped a render/annotation buffer: '+key)
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
        import omni.hydratexture
        self.products={};self.annotators={};self.references={}
        self.frame_info={};self.drawable_observers=[]
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
            # app/Replicator async flags do not express the texture's own
            # render-thread preference. This owner requests a synchronous draw.
            product.hydra_texture.set_async(False)
            # The known-good w_31f/record.py consumes CPU RGB directly. Keep the
            # in-process host copy explicit; the GPU-buffer variant is unverified.
            annotation=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
            annotation.attach(product)
            reference=rep.AnnotatorRegistry.get_annotator('ReferenceTime')
            reference.attach(product)
            self.products[key]=product;self.annotators[key]=annotation;self.references[key]=reference
            def drawable(event,key=key,texture=product.hydra_texture):
                info=texture.get_frame_info(event['result_handle'])
                self.frame_info[key]={name:info.get(name) for name in ('swh_frame_number','frame_number')}
                self.frame_info[key]['view']=list(info['view'])
            self.drawable_observers.append(carb.eventdispatcher.get_eventdispatcher().observe_event(
                event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,on_event=drawable,
                filter=product.hydra_texture.get_event_key(),observer_name='inproc_drawable_'+key))

    def read(self,counts_before):
        images={};cameras={}
        start=time.perf_counter_ns()
        for key,product in self.products.items():
            data=self.annotators[key].get_data()
            data=data.numpy() if hasattr(data,'numpy') else np.asarray(data)
            if data.dtype!=np.uint8 or data.shape!=(480,640,4):
                raise RuntimeError('Playing RGB producer has no complete RGBA frame: '+repr(dict(
                    camera=key,shape=data.shape,dtype=str(data.dtype),
                    compute_count=self.annotators[key].get_node().get_compute_count())))
            info=self.frame_info.get(key,{})
            reference=self.references[key].get_data()
            n,d=reference['referenceTimeNumerator'],reference['referenceTimeDenominator']
            if d<=0:raise RuntimeError('Camera reference clock has no denominator')
            cameras[key]=dict(swh_frame_number=info.get('swh_frame_number'),render_frame_number=info.get('frame_number'),
                frame_identity_source='hydra drawable frame_number',
                render_view=info.get('view'),
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
        self.drawable_observers.clear()

    def diagnostic(self):
        result={}
        for key,annotation in self.annotators.items():
            data=annotation.get_data()
            data=data.numpy() if hasattr(data,'numpy') else np.asarray(data)
            result[key]=dict(shape=list(data.shape),dtype=str(data.dtype),
                compute_count=annotation.get_node().get_compute_count())
        return result

    def graph_diagnostic(self):
        import omni.graph.core as og
        import omni.replicator.core as rep
        import omni.usd
        result=dict(orchestrator_status=str(rep.orchestrator.get_status()),nodes={})
        for prim in omni.usd.get_context().get_stage().Traverse():
            path=str(prim.GetPath())
            if not (path.startswith('/Orchestrator/') or path.startswith('/Render/PostProcess/SDGPipeline/')):continue
            if prim.GetTypeName()!='OmniGraphNode':continue
            node=og.Controller().node(path);values={}
            for attr in node.get_attributes():
                name=attr.get_name()
                if any(token in name.lower() for token in ('referencetime','fabrictime','fabricframetime','rationaltime','simtimes','enabled','step')):
                    try:values[name]=str(attr.get())
                    except Exception as exc:values[name]=repr(exc)
            result['nodes'][path]=dict(type=node.get_type_name(),compute_count=node.get_compute_count(),attributes=values)
        return result


class Playing(Native):
    def __init__(self,scene,manifest,warmup=120):
        import omni.timeline
        self.timeline=omni.timeline.get_timeline_interface()
        self.pending=None;self.frames=FrameLedger();self.priming_reports=[]
        self.prime_on_restore=False
        super().__init__(scene,manifest,cameras=True,use_fabric=True)
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
        self.warmup_rgb=self.rgb.diagnostic()
        self.warmup_graph=self.rgb.graph_diagnostic()
        self.warmup_capture_settings=capture_settings(settings)
        self.pipeline_settings={key:settings.get(key) for key in (
            '/exts/omni.replicator.srtx/enabled','/exts/omni.replicator.core/Orchestrator/enabled',
            '/omni/replicator/asyncRendering','/app/asyncRendering','/rtx/hydra/supportMultiTickRate',
            '/omni/replicator/disableAnnotatorGate','/rtx/rendermode',
            '/physics/updateToUsd','/physics/updateVelocitiesToUsd')}
        self.pipeline_settings['physx_fabric_enabled']=self.world._extension_manager.is_extension_enabled('omni.physx.fabric')
        if (not self.pipeline_settings['physx_fabric_enabled'] or
            self.pipeline_settings['/physics/updateToUsd'] is not False or
            self.pipeline_settings['/physics/updateVelocitiesToUsd'] is not False):
            raise RuntimeError('Playing collector requires the explicit physics-to-Fabric pose path')
        self.pipeline_settings['hydra_texture_async']={key:p.hydra_texture.get_async()
            for key,p in self.rgb.products.items()}
        # The caller restores its own baseline after inspecting initialization.
        # Native.__init__ restores before the physics subscription exists.
        self.prime_on_restore=True

    def create_cameras(self):
        import omni.kit.viewport.utility as vu
        from isaacsim.core.rendering_manager import RenderingManager
        import usdrt
        import omni.usd
        from pxr import UsdGeom
        # Set the history period before render products create Fabric histories.
        RenderingManager.set_dt(1/30)
        RenderingManager._ensure_fabric_simulation_time()
        self.fabric_stage=usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        self.fabric_time=self.fabric_stage.GetPrimAtPath('/ExternalSimulationTime').GetAttribute('omni:time')
        capture_settings(self.settings,enable=True)
        if self.settings.get('/exts/omni.replicator.srtx/enabled') is not False:
            raise RuntimeError('Playing adapter requires the legacy annotator pipeline, as in w_31f/run_arm.py')
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
        # Native pause/play pumps app updates. They must not render a reset pose
        # or advance the sensor clock ahead of the next physical observation.
        self.timeline.set_auto_update(False)
        capture_settings(self.settings,enable=True)
        for product in self.rgb.products.values():product.hydra_texture.set_updates_enabled(False)
        try:
            # This synchronous owner never pumps an app update during restore.
            # Pause/play would reset the sensor scheduler and discard its frame.
            result=super().restore(baseline,manage_timeline=False)
            # An episode's final cancellation may already have paused the World.
            # Commit play directly; World.play() would pump another app update.
            if not self.timeline.is_playing():
                self.timeline.play()
                self.timeline.commit()
            self.align_timeline()
        finally:
            self.timeline.set_auto_update(True)
            for product in self.rgb.products.values():product.hydra_texture.set_updates_enabled(True)
        self.pending=None
        self.frames.reset(result['epoch'],result['invalid_through_ns'],result['snapshot']['physics_time_s'])
        return self.prime(result) if self.prime_on_restore else result

    def prime(self,restored,max_updates=30,required=3):
        """Discard a bounded startup transient while keeping restored drives."""
        if not 3<=required<=max_updates<=30:raise ValueError('Invalid priming bound')
        report=dict(epoch=restored['epoch'],max_updates=max_updates,required=required,
            passed=False,updates=[],consecutive=0)
        self.priming_reports.append(report)
        candidate=FrameLedger();previous=self.clock()
        targets=restored['command_baseline']['drive_targets']
        started=time.perf_counter_ns()
        for index in range(max_updates):
            packet,counts,_=self.advance()
            snapshot=self.snapshot()
            if (snapshot['epoch']!=restored['epoch'] or snapshot['sim_ns']!=packet['sim_ns'] or
                not matches(snapshot['drive_targets'],targets,'target_rad') or self.last_apply is not None):
                raise RuntimeError('Priming changed the restored command baseline')
            row=dict(index=index,packet=copy.deepcopy(packet),accepted=False,
                drive_targets=list(snapshot['drive_targets']))
            report['updates'].append(row)
            counts_now=list(packet['render_completions'].values())
            valid=counts_now==[1,1]
            if valid:
                try:
                    _,cameras,_=self.rgb.read(counts)
                    packet.update(cameras=cameras,observation_epoch=previous['epoch'],
                        observation_ns=previous['sim_ns'],image_delay_updates=1)
                    if candidate.previous is None:
                        candidate.reset(previous['epoch'],previous['sim_ns'],previous['physics_time_s'])
                    candidate.accept(packet)
                except RuntimeError as exc:
                    valid=False;row['camera_error']=str(exc)
            row['packet']=copy.deepcopy(packet)
            if valid:report['consecutive']+=1
            else:report['consecutive']=0;candidate=FrameLedger()
            previous=snapshot
            if report['consecutive']>=required:
                report.update(passed=True,wall_ms=(time.perf_counter_ns()-started)/1e6,
                    final_sim_ns=snapshot['sim_ns'])
                self.pending=None;self.floor=snapshot['sim_ns']
                self.frames.reset(snapshot['epoch'],self.floor,snapshot['physics_time_s'],packet['cameras'])
                return dict(epoch=snapshot['epoch'],invalid_through_ns=self.floor,
                    arm_restored=restored['arm_restored'],snapshot=snapshot,
                    command_baseline=dict(epoch=snapshot['epoch'],sim_ns=self.floor,
                        physics_time_s=snapshot['physics_time_s'],drive_targets=list(targets)),
                    restore_verification=copy.deepcopy(restored),priming=copy.deepcopy(report))
        report['wall_ms']=(time.perf_counter_ns()-started)/1e6
        raise RuntimeError('Camera priming did not stabilize within '+str(max_updates)+' updates')

    def align_timeline(self):
        # World play/pause pumps updates without physics, and float32 PhysX dt
        # accumulates differently from the timeline's 1/30. Own their alignment
        # without another app update; never rewrite the camera's original stamp.
        native=self.native_time()
        if not matches(round(self.timeline.get_current_time()*1e9),round(native*1e9),'native_ns'):
            self.timeline.set_current_time(native)
            self.timeline.commit_silently()

    def native_time(self):
        value=self.fabric_time.Get()
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError('Fabric renderer clock is unavailable or non-finite')
        return float(value)

    def stamp(self):
        # Read the native graph clock used by cameras. World accumulates float32
        # PhysX callbacks independently; the shared contract checks its deltas.
        return round(self.native_time()*1e9)

    def advance(self):
        """One measured physical/render update, shared by priming and recording."""
        if not self.world.is_playing():raise RuntimeError('Playing collector timeline unexpectedly stopped')
        flags=capture_settings(self.settings)
        observation=self.clock()
        self.align_timeline()
        start=time.perf_counter_ns();old_steps=self.actual_physics_steps
        index=self.world.current_time_step_index;before=dict(self.rgb.render_events)
        fabric_sync_ms=flush_physics_to_fabric(self.world)
        if self.clock()!=observation or self.actual_physics_steps!=old_steps or dict(self.rgb.render_events)!=before:
            raise RuntimeError('Fabric publication changed the native clock or rendered a frame')
        self.world.step(render=True)
        update_end=time.perf_counter_ns()
        timeline_ns=round(self.timeline.get_current_time()*1e9)
        snapshot=self.clock()
        self.last_frame=dict(epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],timeline_ns=timeline_ns,
            capture_settings=flags,fabric_sync_ms=fabric_sync_ms,
            render_completions={p.path:self.rgb.render_events[p.path]-before.get(p.path,0)
                for p in self.rgb.products.values()},
            physics_time_s=snapshot['physics_time_s'],native_physics_steps=self.actual_physics_steps-old_steps,
            world_step_count=self.world.current_time_step_index-index)
        fabric_time=self.fabric_time.Get()
        if fabric_time is None or not math.isfinite(float(fabric_time)):
            raise RuntimeError('Fabric renderer clock is unavailable or non-finite')
        self.last_frame['fabric_time_s']=float(fabric_time)
        self.frames.check_advance(self.last_frame,observation)
        return self.last_frame,before,(update_end-start)/1e6

    def step(self):
        if self.pending is not None:raise RuntimeError('Previous playing frame has not been consumed')
        observation=self.clock()
        recording=self.last_apply==(observation['epoch'],observation['sim_ns'])
        _,before,update_ms=self.advance()
        snapshot=self.clock();start=time.perf_counter_ns()
        completions=list(self.last_frame['render_completions'].values())
        if any(count!=1 for count in completions):raise RuntimeError('Expected exactly one new render for both cameras')
        images,cameras,copy_ms=self.rgb.read(before)
        matches_clock=[(stamp,delay) for stamp,delay in ((snapshot['sim_ns'],0),(observation['sim_ns'],1))
            if all(matches(v['stamp_ns'],stamp,'stamp_ns') for v in cameras.values())]
        if len(matches_clock)!=1:raise RuntimeError('Camera clocks are outside the bounded native observation window')
        target,delay=matches_clock[0]
        packet=dict(self.last_frame,cameras=cameras,observation_epoch=observation['epoch'],
            observation_ns=target,image_delay_updates=delay);self.last_frame=packet
        self.frames.accept(packet)
        if not recording and target<=self.floor+1000:
            # The first observation remains one tick after the measured boundary.
            # Even this discarded boundary image must pass the strict ledger.
            self.frames.reset(snapshot['epoch'],snapshot['sim_ns'],snapshot['physics_time_s'],cameras)
            return update_ms+(time.perf_counter_ns()-start)/1e6
        profile=dict(app_update_ms=update_ms-packet['fabric_sync_ms'],fabric_sync_ms=packet['fabric_sync_ms'],camera_read_ms=copy_ms,
            camera_stamps={k:v['stamp_ns'] for k,v in cameras.items()},playing_frame=packet,
            image_delay_updates=delay,
            synchronization='native stamp join, zero or one app update; drawable and copy counters')
        self.pending=(images,profile)
        return update_ms+(time.perf_counter_ns()-start)/1e6

    def frame_ready(self,expected):
        if self.pending is None:return False
        packet=self.pending[1]['playing_frame']
        if expected['epoch']!=packet['observation_epoch'] or expected['sim_ns']!=packet['observation_ns']:
            raise RuntimeError('An unconsumed camera frame belongs to another observation')
        return True

    def render(self,expected=None):
        if self.pending is None:raise RuntimeError('No new playing frame is ready')
        packet=self.pending[1]['playing_frame']
        if packet['epoch']!=self.epoch or packet['sim_ns']!=self.stamp():raise RuntimeError('Playing RGB cache is stale')
        if expected is None or expected['epoch']!=packet['observation_epoch'] or expected['sim_ns']!=packet['observation_ns']:
            raise RuntimeError('Image join requested another observation')
        result=self.pending;self.pending=None
        return result

    def trace_steps(self,count=8):
        """Fault diagnostics only: never produces accepted rows or commands."""
        rows=[]
        for index in range(count):
            before=self.snapshot();counts=dict(self.rgb.render_events)
            old_steps=self.actual_physics_steps;old_index=self.world.current_time_step_index
            self.align_timeline();self.world.step(render=True);after=self.snapshot()
            row=dict(index=index,before_ns=before['sim_ns'],after_ns=after['sim_ns'],
                before_state=before['state'],after_state=after['state'],
                physical_steps=self.actual_physics_steps-old_steps,world_steps=self.world.current_time_step_index-old_index,
                render_completions={p.path:self.rgb.render_events[p.path]-counts.get(p.path,0)
                    for p in self.rgb.products.values()},accepted=False)
            try:
                _,cameras,_=self.rgb.read(counts);row['cameras']=cameras
            except Exception as exc:row['read_error']=repr(exc)
            rows.append(row)
        return rows

    def close(self):
        self.physics_subscription=None
        super().close()
