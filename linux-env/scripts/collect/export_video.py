"""CPU decoded-frame/PTS equivalence of raw episode video and dataset video."""
import hashlib,math,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from export_content import ContentMismatch,read_parquet_rows

POLICY=dict(version=1,lossless='exact RGB equality',lossy_min_psnr_db=32.,lossy_max_abs_rgb_error=224,
    rationale='H264 yuv420 re-encoding: initial full-episode calibration min35.34dB and peak176;3.34dB MSE margin,48-level isolated-edge margin',
    alignment='exact frame ordinal plus PTS within one encoded timebase quantum; episode offsets/counts from raw lengths',
    limitation='Lossy equivalence is bounded, not pixel identity; differences within these bounds are indistinguishable')
LOSSLESS={'ffv1','png','rawvideo','huffyuv','qtrle'}


def compare_rgb(raw,stored,lossless):
    if raw.shape!=stored.shape or raw.dtype!=np.uint8 or stored.dtype!=np.uint8:
        raise ContentMismatch('video layout',dict(raw_shape=raw.shape,stored_shape=stored.shape))
    difference=raw.astype(np.float32)-stored.astype(np.float32)
    max_error=int(np.max(np.abs(difference)));mse=float(np.mean(difference*difference,dtype=np.float64))
    psnr=math.inf if mse==0 else 10*math.log10(255.**2/mse)
    passed=max_error==0 if lossless else psnr>=POLICY['lossy_min_psnr_db'] and max_error<=POLICY['lossy_max_abs_rgb_error']
    if not passed:raise ContentMismatch('video content',dict(psnr_db=None if math.isinf(psnr) else psnr,max_abs_rgb_error=max_error,lossless=lossless,policy=POLICY))
    return psnr,max_error


class VideoFrames:
    def __init__(self,path):
        import av
        self.path=Path(path);self.container=av.open(str(path));self.stream=self.container.streams.video[0]
        self.stream.codec_context.thread_count=1
        self.time_base=Fraction(self.stream.time_base)
        if self.time_base<=0:raise ContentMismatch('video clock',dict(path=str(path),reason='Invalid timebase'))
        self.codec=self.stream.codec_context.name;self.count=self.stream.frames
    def frames(self,start,count,fps):
        target=Fraction(start,fps);self.container.seek(int(target/self.time_base),stream=self.stream,backward=True)
        found=0
        for frame in self.container.decode(self.stream):
            if frame.pts is None:raise ContentMismatch('video PTS',dict(path=str(self.path),reason='Missing PTS'))
            stamp=Fraction(frame.pts)*self.time_base
            if stamp<target-self.time_base:continue
            expected=Fraction(start+found,fps)
            if abs(stamp-expected)>self.time_base:
                raise ContentMismatch('video frame order/PTS',dict(path=str(self.path),frame=start+found,actual_s=float(stamp),expected_s=float(expected),quantum_s=float(self.time_base)))
            yield frame.to_ndarray(format='rgb24')
            found+=1
            if found==count:return
        raise ContentMismatch('video frame count',dict(path=str(self.path),expected=count,decoded=found,start=start))
    def close(self):self.container.close()


def selected_episodes(run,details,count):
    if count is None or count>=len(details):return list(range(len(details)))
    # Stable, distributed choices with two different task colours where possible.
    seed=int(hashlib.sha256(str(Path(run).parent.name).encode()).hexdigest()[:12],16)
    order=sorted(range(len(details)),key=lambda i:hashlib.sha256((str(seed)+':'+str(i)).encode()).digest())
    chosen=[order[0]]
    for i in order[1:]:
        if details[i]['task']!=details[chosen[0]]['task']:chosen.append(i);break
    for i in order:
        if len(chosen)>=count:break
        if i not in chosen:chosen.append(i)
    return sorted(chosen[:count])


def validate_videos(run,metadata,sample_count=2):
    import json
    started=time.monotonic();run=Path(run);root=run/'dataset'
    if not metadata.get('passed'):raise ContentMismatch('video prerequisite',dict(reason='Metadata has not passed'))
    info=json.loads((root/'meta/info.json').read_text());fps=info['fps'];details=metadata['episode_details']
    cols=['episode_index','length']+[f'videos/observation.images.{c}/{k}' for c in ('top','wrist') for k in ('chunk_index','file_index','from_timestamp','to_timestamp')]
    episodes=read_parquet_rows(sorted((root/'meta/episodes').rglob('*.parquet')),cols)
    selected=selected_episodes(run,details,sample_count);locations={};file_counts={}
    for i,ep in enumerate(episodes):
        if ep['episode_index']!=i or ep['length']!=details[i]['frames']:raise ContentMismatch('video episode map',dict(episode=i))
        for camera in ('top','wrist'):
            key='observation.images.'+camera;prefix='videos/'+key+'/'
            path=root/info['video_path'].format(video_key=key,chunk_index=ep[prefix+'chunk_index'],file_index=ep[prefix+'file_index'])
            if not path.resolve().is_relative_to(root.resolve()):raise ContentMismatch('video path',dict(path=str(path)))
            start=file_counts.get(str(path),0);end=start+ep['length']
            for name,expected in (('from_timestamp',start/fps),('to_timestamp',end/fps)):
                if abs(ep[prefix+name]-expected)>1e-6:raise ContentMismatch('video episode offset',dict(episode=i,camera=camera,field=name,actual=ep[prefix+name],expected=expected))
            locations[i,camera]=(path,start);file_counts[str(path)]=end
    # Container sample counts cover all files; pixel comparison covers selected episodes.
    for path,count in file_counts.items():
        video=VideoFrames(path)
        try:
            if video.count!=count:raise ContentMismatch('video container count',dict(path=path,actual=video.count,expected=count))
        finally:video.close()
    reports=[];total=0
    for episode in selected:
        detail=details[episode];n=detail['frames'];result=dict(episode_index=episode,raw_attempt=detail['raw_attempt'],seed=detail['seed'],cameras={})
        for camera in ('top','wrist'):
            raw_path=run/detail['raw_attempt']/(camera+'.mp4');path,start=locations[episode,camera]
            raw=VideoFrames(raw_path);stored=VideoFrames(path);source_hash=hashlib.sha256();stored_hash=hashlib.sha256()
            try:
                if raw.count!=n:raise ContentMismatch('raw video count',dict(episode=episode,camera=camera,actual=raw.count,expected=n))
                worst_psnr=math.inf;worst_error=0;checked=0
                for index,(a,b) in enumerate(zip(raw.frames(0,n,fps),stored.frames(start,n,fps))):
                    if a.shape!=(480,640,3) or b.shape!=(480,640,3):raise ContentMismatch('deployment video shape',dict(episode=episode,camera=camera,frame=index))
                    try:psnr,error=compare_rgb(a,b,stored.codec in LOSSLESS)
                    except ContentMismatch as exc:
                        exc.details.update(episode=episode,camera=camera,frame=index,raw_path=str(raw_path),stored_path=str(path));raise
                    worst_psnr=min(worst_psnr,psnr);worst_error=max(worst_error,error);checked+=1
                    source_hash.update(a.tobytes());stored_hash.update(b.tobytes())
                if checked!=n:raise ContentMismatch('decoded frame count',dict(episode=episode,camera=camera,actual=checked,expected=n))
                result['cameras'][camera]=dict(passed=True,frames=checked,codec=stored.codec,min_psnr_db=None if math.isinf(worst_psnr) else worst_psnr,
                    max_abs_rgb_error=worst_error,ordered_raw_rgb_sha256=source_hash.hexdigest(),ordered_stored_rgb_sha256=stored_hash.hexdigest(),
                    timebase_s=float(stored.time_base),stored_start_frame=start)
                total+=checked
            finally:raw.close();stored.close()
        reports.append(result)
    return dict(passed=True,scope='ALL frames' if sample_count is None else 'Two full sampled episodes per batch',policy=POLICY,
        selected_episodes=selected,episodes=reports,decoded_frame_pairs=total,all_video_container_counts_checked=True,wall_s=time.monotonic()-started)
