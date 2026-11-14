"""Two RGB producers synchronized by Replicator's writer execution barrier."""
import time
import numpy as np


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
                outer.images=payload_images(data,names,outer.expected)
                outer.calls+=1
                outer.copy_ms+=(time.perf_counter_ns()-start)/1e6
        self.images=None;self.expected=None;self.calls=0;self.copy_ms=0.
        self.writer=BarrierWriter()
        # Installed writers.py connects EACH cloned RGB producer's exec output
        # to OgnRationalTimeSyncGateOnce before the callback can consume arrays.
        self.writer.attach(list(self.products.values()))
        rep.orchestrator.set_capture_on_play(False)

    def read(self,seconds):
        import omni.replicator.core as rep
        self.expected=seconds;self.images=None;self.calls=0;self.copy_ms=0.
        start=time.perf_counter_ns()
        try:
            rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=False,wait_for_render=True)
            if self.calls!=1 or self.images is None:raise RuntimeError('RGB barrier did not deliver exactly one paired frame')
            return self.images,dict(render_wait_ms=(time.perf_counter_ns()-start)/1e6-self.copy_ms,
                camera_read_ms=self.copy_ms,reference_time_s=seconds,
                synchronization='per-product RGB execution -> rational-time writer barrier')
        finally:self.expected=None

    def close(self):
        self.writer.detach()
        for product in self.products.values():product.destroy()
