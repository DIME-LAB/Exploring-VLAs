"""Opt-in CPU diagnostic: causal two-stage H.264 with marked held-image filler.

The offline training encoder uses future frames. This adapter supplies copies of
the current held render as lookahead, without advancing physics or returning an
older image. It preserves encoder/decoder state, but is an approximation to the
training movie's temporal context. It is not a deployment recommendation.
"""
from fractions import Fraction
import hashlib
import time
import av
import numpy as np

class Stage:
    def __init__(self, options, interpolation=None):
        self.options=dict(options)
        self.interpolation=interpolation
        self.encoder=av.CodecContext.create('libx264','w')
        self.encoder.width=640;self.encoder.height=480
        self.encoder.pix_fmt='yuv420p';self.encoder.time_base=Fraction(1,30)
        self.encoder.framerate=Fraction(30,1);self.encoder.thread_count=1
        self.encoder.options=options
        self.decoder=av.CodecContext.create('h264','r');self.decoder.thread_count=1
    def feed(self, rgb, index):
        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
        if self.interpolation is not None:frame=frame.reformat(format='yuv420p',interpolation=self.interpolation)
        frame.pts=index;frame.time_base=Fraction(1,30)
        frames=[]
        for packet in self.encoder.encode(frame):frames.extend(self.decoder.decode(packet))
        return frames

class CameraStream:
    def __init__(self):
        # ffmpeg's automatic RGB->420 conversion is bicubic; PyAV's default
        # conversion is bilinear. The offline export itself uses PyAV defaults.
        self.recorder=Stage({'preset':'ultrafast','crf':'18','threads':'1'},'BICUBIC')
        self.exporter=Stage({'g':'2','crf':'30','threads':'1'})
        self.index=0;self.real_frames=0;self.filler_frames=0
    def transform(self, image):
        assert image.dtype==np.uint8 and image.shape==(480,640,3)
        assert image.flags.c_contiguous
        start=time.monotonic();requested=self.index;self.real_frames+=1
        fillers=0
        # Default x264 lookahead is finite. Fail closed rather than serving an
        # old frame, silently flushing the stream, or advancing simulation.
        for attempt in range(128):
            index=self.index;self.index+=1
            if attempt:fillers+=1;self.filler_frames+=1
            stage1=self.recorder.feed(image,index)
            for recorded in stage1:
                assert recorded.pts<=index and index-recorded.pts<8,'Collector codec unexpectedly delayed a frame'
                for frame in self.exporter.feed(recorded.to_ndarray(format='rgb24'),recorded.pts):
                    assert frame.pts<=requested,'Skipped requested codec output'
                    if frame.pts==requested:
                        output=frame.to_ndarray(format='rgb24')
                        return output,dict(mode='running-two-stage-held-lookahead-diagnostic',requested_codec_index=requested,returned_codec_index=int(frame.pts),key_frame=bool(frame.key_frame),pict_type=int(frame.pict_type),held_filler_frames=fillers,total_held_filler_frames=self.filler_frames,real_capture_count=self.real_frames,wall_s=time.monotonic()-start,raw_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),encoded_rgb_sha256=hashlib.sha256(output.tobytes()).hexdigest(),recorder_options=self.recorder.options,exporter_options=self.exporter.options)
        raise TimeoutError('H.264 current-frame output exceeded128held fillers')

class EvalCodec:
    def __init__(self):self.streams={camera:CameraStream() for camera in ('top','wrist')}
    def transform(self, observation):
        result=dict(observation);metadata={}
        for camera,stream in self.streams.items():result[camera],metadata[camera]=stream.transform(np.ascontiguousarray(observation[camera]))
        return result,metadata
