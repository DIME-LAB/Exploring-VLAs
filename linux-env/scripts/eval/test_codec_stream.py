"""CPU integration controls against the actual ffmpeg + export codec pipeline."""
from pathlib import Path
import argparse,json,os,subprocess
import numpy as np
from PIL import Image
from torchcodec.decoders import VideoDecoder
import torch
from codec_stream import CameraStream
from codec_pair import cv_frames,export_encode

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available() and not torch.cuda.is_initialized()
    torch.set_num_threads(4);assert not a.out.exists();a.out.mkdir()
    rows=[]
    for camera in ('top','wrist'):
        stream=CameraStream();images=[];outputs=[];events=[]
        for seed in (530000,530001,530002):
            image=np.array(Image.open(a.input/f'seed-{seed}'/f'{camera}-raw_live.png'));before=image.copy()
            output,event=stream.transform(image);assert np.array_equal(image,before)
            assert event['returned_codec_index']==event['requested_codec_index']==len(images)
            images.extend([image]*(event['held_filler_frames']+1));outputs.append(output);events.append(event)
        # Recreate exactly the real+held-filler input sequence using the original
        # recorder executable and the production export AV loop, then training's
        # decoder. This checks current-frame identity through changing inputs.
        recorded=a.out/f'{camera}-collector.mp4'
        command=['/usr/bin/ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','30','-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','18','-threads','1','-pix_fmt','yuv420p',str(recorded)]
        subprocess.run(command,input=b''.join(image.tobytes() for image in images),check=True)
        exported=a.out/f'{camera}-export.mp4';export_encode(cv_frames(recorded),exported)
        decoder=VideoDecoder(str(exported),device='cpu',num_ffmpeg_threads=1)
        frames=decoder.get_frames_at(indices=[event['requested_codec_index'] for event in events]).data.permute(0,2,3,1).numpy()
        errors=[float(np.mean(abs(output.astype(float)-frame.astype(float)))) for output,frame in zip(outputs,frames)]
        assert errors==[0.,0.,0.],errors
        # A fresh episode must recreate the first output exactly.
        reset,reset_event=CameraStream().transform(images[0]);assert np.array_equal(reset,outputs[0]) and reset_event['requested_codec_index']==0
        rows.append(dict(camera=camera,events=events,offline_training_decoder_mae_dn=errors,episode_reset_exact=True,inputs_unmodified=True))
    proof=dict(status='PASS',cuda_initialized=torch.cuda.is_initialized(),controls=rows,scope='Same generated causal stream is pixel-identical to recorder+export+TorchCodec; future true scene context remains approximated by marked held copies')
    (a.out/'result.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
