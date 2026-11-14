"""Two RGB producers synchronized by Replicator's writer execution barrier."""
import time
import numpy as np
from collections import Counter


def payload_images(data,names,expected_s):
    numerator,denominator=data['reference_time']
    if denominator<=0 or abs(numerator/denominator-expected_s)>1e-6:
        raise RuntimeError('RGB writer reference time differs from physical observation')
    images={}
    for key,name in names.items():
        value=data['rgb-'+name]
        if value.dtype!=np.uint8 or value.shape!=(480,640,4):
            raise RuntimeError('Unexpected RGB writer layout: '+key)
        images[key]=np.ascontiguousarray(value[:,:,:3])
    if set(images)!={'wrist','top'}:raise RuntimeError('Both RGB producers are required')
    return images


class RGB:
    def __init__(self,paths):
        import omni.replicator.core as rep
        import omni.usd
        import carb.eventdispatcher
        import omni.syntheticdata._syntheticdata as syntheticdata
        interface=syntheticdata.acquire_syntheticdata_interface()
        self.render_events=Counter();self.event_samples=[]
        def completion(event):
            payload=interface.parse_rendered_simulation_event(event['product_path_handle'],event['results'])
            key=str(payload[0])
            self.render_events[str(key)]+=1
            if len(self.event_samples)<4:self.event_samples.append({k:str(v) for k,v in event.payload.items()})
        self.observer=carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME,True),
            on_event=completion,observer_name='inproc_render_counter')
        self.products={key:rep.create.render_product(path,(640,480)) for key,path in paths.items()}
        names={key:product.path.rsplit('/',1)[-1] for key,product in self.products.items()}
        if len(set(names.values()))!=2:raise RuntimeError('Render products must be distinct')
        outer=self
        class BarrierWriter(rep.Writer):
            def __init__(self):
                self.annotators=[rep.AnnotatorRegistry.get_annotator('rgb',device='cpu',do_array_copy=True)]
            def write(self,data):
                if outer.expected is None:return
                start=time.perf_counter_ns()
                outer.calls+=1
                outer.reference_time=tuple(map(int,data['reference_time']))
                try:outer.images=payload_images(data,names,outer.expected)
                except BaseException as exc:outer.error=exc
                outer.copy_ms+=(time.perf_counter_ns()-start)/1e6
        self.images=None;self.expected=None;self.calls=0;self.copy_ms=0.
        self.writer=BarrierWriter()
        # Installed writers.py connects EACH cloned RGB producer's exec output
        # to OgnRationalTimeSyncGateOnce before the callback can consume arrays.
        self.writer.attach(list(self.products.values()))
        rep.orchestrator.preview()
        rep.orchestrator.set_capture_on_play(True)
        import carb
        carb.settings.get_settings().set('/omni/replicator/asyncRendering',False)
        carb.settings.get_settings().set('/rtx/hydra/supportMultiTickRate',True)
        self.enable(False)

    def enable(self,enabled):
        for product in self.products.values():product.hydra_texture.set_updates_enabled(enabled)

    def start_capture(self):
        import omni.replicator.core as rep
        from omni.kit.async_engine import run_coroutine
        return run_coroutine(rep.orchestrator.step_async(rt_subframes=1,delta_time=0.,
            pause_timeline=False,wait_for_render=True))

    def read(self,seconds,render_once):
        self.expected=seconds;self.images=None;self.calls=0;self.copy_ms=0.
        self.error=None;self.reference_time=None
        start=time.perf_counter_ns()
        task=None
        try:
            self.enable(True)
            task=self.start_capture()
            updates=0
            deadline=time.monotonic()+20.
            while not task.done() and time.monotonic()<deadline:
                render_once();updates+=1
            if not task.done():raise TimeoutError('Held RGB capture exceeded 20 wall seconds')
            task.result()
            if self.error:raise RuntimeError('RGB producer failed at '+repr(self.reference_time)) from self.error
            if self.calls!=1 or self.images is None:
                nodes=[dict(name=a.name,count=a.get_node().get_compute_count(),path=str(a.get_node().get_prim_path()))
                    for a in self.writer._attached_annotators]
                raise RuntimeError('RGB barrier did not deliver exactly one paired frame: '+repr(dict(calls=self.calls,updates=updates,events=dict(self.render_events),nodes=nodes)))
            return self.images,dict(render_wait_ms=(time.perf_counter_ns()-start)/1e6-self.copy_ms,
                camera_read_ms=self.copy_ms,reference_time_s=seconds,
                camera_stamps={key:round(self.reference_time[0]/self.reference_time[1]*1e9)
                    for key in self.products},
                reference_time=list(self.reference_time),render_app_updates=updates,
                synchronization='per-product RGB execution -> rational-time writer barrier')
        finally:
            if task is not None and not task.done():task.cancel()
            self.expected=None
            self.enable(False)

    def close(self):
        self.enable(False)
        self.writer.detach()
        for product in self.products.values():product.destroy()
        self.observer=None
