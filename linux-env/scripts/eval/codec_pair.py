"""CPU-only matched-home codec diagnostic; never changes deployment defaults."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, subprocess, time
import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

def export_encode(frames, path):
    # Identical frame conversion and codec options to pinned video_utils.py;
    # the intermediate PNG/PIL round trip is lossless for uint8 RGB.
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('h264', rate=30, options={'g':'2','crf':'30','threads':'1'})
        stream.width=640; stream.height=480; stream.pix_fmt='yuv420p'
        for rgb in frames:
            frame=av.VideoFrame.from_image(Image.fromarray(rgb).convert('RGB'))
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    with av.open(str(path)) as decoded:
        frame=next(decoded.decode(video=0))
        return frame.to_ndarray(format='rgb24'),dict(key_frame=frame.key_frame,pict_type=str(frame.pict_type),codec=decoded.streams.video[0].codec_context.name)

def cv_frames(path):
    capture=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,bgr=capture.read()
            if not ok: break
            yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally: capture.release()

def metrics(a,b):
    delta=a.astype(float)-b.astype(float);mse=float(np.mean(delta**2))
    return dict(psnr_db=float(10*np.log10(255**2/mse)) if mse else None,mae_dn=float(np.mean(abs(delta))),exact=bool(np.array_equal(a,b)))

def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','checkpoint','native-root','teacher-root','source-root','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);pa.set_cpu_count(4);pa.set_io_thread_count(1);cv2.setNumThreads(1)
    config=PreTrainedConfig.from_pretrained(a.checkpoint);config.device='cpu';config.compile_model=False;config.rtc_config=None
    policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cpu').eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=a.checkpoint,preprocessor_overrides={'device_processor':{'device':'cpu'},'rename_observations_processor':{'rename_map':{}}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
    root=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('codec_fk',root/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py');fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
    ds=LeRobotDataset('local/smolvla-phase3-normal-60k-s42-20261114',root=a.dataset,episodes=list(range(5)),video_backend='torchcodec',delta_timestamps={'action':[i/30 for i in range(50)]},return_uint8=False)
    table=pq.read_table(sorted(a.dataset.glob('data/**/*.parquet')),columns=['index','episode_index','frame_index']).to_pydict()
    indices={ep:index for ep,frame,index in zip(table['episode_index'],table['frame_index'],table['index']) if ep<5 and frame==0}
    episodes=pq.read_table(sorted(a.dataset.glob('meta/episodes/**/*.parquet'))).to_pylist()
    def predict(batch):
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        policy.reset();torch.manual_seed(42)
        with torch.inference_mode(): result=post(policy.predict_action_chunk(pre(batch)).flatten(0,1)).reshape(1,50,6)[0].float().numpy()
        assert hashes=={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        return result
    rows=[];started=time.monotonic()
    for ep in range(5):
        seed=530000+ep;folder=a.out/f'seed-{seed}';folder.mkdir();batch=default_collate([ds[int(indices[ep])]])
        baseline=predict(batch);assert abs(baseline-np.load(a.teacher_root/f'seed-{seed}.npz')['prediction'][0]).max()<1e-5
        tcp=fk.forward_kinematics_batch(np.radians(baseline[:,:5]));live=np.load(a.native_root/f'ep{ep:02d}'/'first-cameras.npz')
        variants={name:dict(batch) for name in ('raw_live','live_export_only','live_both_encodes','source_decoded','source_export_control')};pixels={};gop={}
        for camera in ('top','wrist'):
            key='observation.images.'+camera;recorded=(batch[key][0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
            actual=live[camera];assert actual.shape==(480,640,3)
            source=a.source_root/f'attempt-{ep:04d}'/(camera+'.mp4');assert source.exists()
            source_first=next(cv_frames(source))
            control,info=export_encode(cv_frames(source),folder/f'{camera}-source-export.mp4')
            single,info_single=export_encode((actual for _ in range(60)),folder/f'{camera}-live-export.mp4')
            stage1=folder/f'{camera}-live-collector.mp4'
            command=['/usr/bin/ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','30','-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','18','-threads','1','-pix_fmt','yuv420p',str(stage1)]
            subprocess.run(command,input=actual.tobytes()*60,check=True)
            both,info_both=export_encode(cv_frames(stage1),folder/f'{camera}-live-both.mp4')
            meta=episodes[ep];prefix='videos/'+key;timestamp=meta[prefix+'/from_timestamp']
            video=a.dataset/f'videos/{key}/chunk-{meta[prefix+"/chunk_index"]:03d}/file-{meta[prefix+"/file_index"]:03d}.mp4'
            with av.open(str(video)) as container:
                stream=container.streams.video[0];container.seek(int(timestamp/stream.time_base),stream=stream,backward=True)
                frame=next(f for f in container.decode(stream) if abs(float(f.time)-timestamp)<.01)
                gop[camera]=dict(dataset_timestamp=timestamp,dataset_key_frame=frame.key_frame,dataset_pict_type=str(frame.pict_type),source_control=info,live_single=info_single,live_both=info_both)
            values=dict(raw_live=actual,live_export_only=single,live_both_encodes=both,source_decoded=source_first,source_export_control=control)
            pixels[camera]={name:metrics(value,recorded) for name,value in values.items()}
            Image.fromarray(recorded).save(folder/f'{camera}-dataset.png')
            for name,value in values.items():
                Image.fromarray(value).save(folder/f'{camera}-{name}.png');variants[name][key]=torch.from_numpy(value.copy()).permute(2,0,1)[None].float()/255
        predictions={};effects={}
        for name,variant in variants.items():
            prediction=predict(variant);predictions[name]=prediction
            distance=np.linalg.norm(fk.forward_kinematics_batch(np.radians(prediction[:,:5]))-tcp,axis=1)*1000
            effects[name]=dict(tcp_mean_mm=float(distance.mean()),tcp_max_mm=float(distance.max()),tcp_first10_mean_mm=float(distance[:10].mean()),arm_l1_deg=float(abs(prediction-baseline)[:,:5].mean()))
        row=dict(seed=seed,fixed_noise_seed=42,gop=gop,pixels=pixels,prediction_shift_vs_dataset=effects,context_caveat='Live codec tests repeat one exact raw home frame for 60 frames; source positive control encodes the complete recorded episode. Original pre-encoding RGB unavailable; raw_live is native exact-state re-render fallback.')
        rows.append(row);np.savez_compressed(folder/'predictions.npz',dataset=baseline,**predictions);(a.out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    receipt=dict(status='complete',wall_s=time.monotonic()-started,cuda_available=torch.cuda.is_available(),cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads(),nice=os.getpriority(os.PRIO_PROCESS,0),opencv=cv2.__version__,pyav=av.__version__,libav={k:list(v) for k,v in av.library_versions.items()},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),export_options=dict(codec='h264/libx264',g=2,crf=30,threads=1,pix_fmt='yuv420p',preset='default'),collector_options=dict(codec='libx264',preset='ultrafast',crf=18,threads=1,pix_fmt='yuv420p'))
    assert not receipt['cuda_initialized'];(a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
