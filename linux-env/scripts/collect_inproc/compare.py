"""Compare same-seed trajectories, synchronized video pairs and stage timings."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def image_metrics(a,b):
    import cv2
    a,b=a.astype(np.float64),b.astype(np.float64)
    mse=float(np.mean((a-b)**2));mae=float(np.mean(abs(a-b)))
    blur=lambda value:cv2.GaussianBlur(value,(11,11),1.5)
    ma,mb=blur(a),blur(b);va=blur(a*a)-ma*ma;vb=blur(b*b)-mb*mb
    cov=blur(a*b)-ma*mb
    ssim=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    return dict(psnr_db=10*math.log10(255**2/mse) if mse else None,
        identical=mse==0,ssim=float(np.mean(ssim[5:-5,5:-5])),mae_rgb=mae)


def stage_summary(profiles):
    keys={k for p in profiles for k,v in p.items() if k.endswith('_ms') and isinstance(v,(int,float))}
    return {k:dict(total_ms=sum(p.get(k,0.) for p in profiles),
        median_ms=float(np.median([p.get(k,0.) for p in profiles])),
        p95_ms=float(np.percentile([p.get(k,0.) for p in profiles],95))) for k in sorted(keys)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay',action='store_true',help='Require identical original physical rows for a renderer comparison')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import cv2
    def rows(path):return [json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
    left,right=rows(a.reference),rows(a.candidate)
    if not left or not right:raise ValueError('Empty comparison')
    if a.replay and left!=right:raise ValueError('Renderer comparison requires identical physical rows')
    if not a.replay:
        lr=json.loads((a.reference/'receipt.json').read_text());rr=json.loads((a.candidate/'receipt.json').read_text())
        if any(lr['params'][k]!=rr['params'][k] for k in ('seed','color','object')):
            raise ValueError('Comparison seeds/targets differ')
    sample_indexes=sorted({0,min(len(left),len(right))//2,min(len(left),len(right))-1})
    metrics={};sheets=[]
    for camera in ('wrist','top'):
        captures=[cv2.VideoCapture(str(path/(camera+'.mp4'))) for path in (a.reference,a.candidate)]
        entries=[]
        try:
            for index in range(min(len(left),len(right))):
                frames=[capture.read() for capture in captures]
                if not all(ok for ok,frame in frames):raise ValueError('Missing video frame')
                if index not in sample_indexes:continue
                images=[frame for ok,frame in frames]
                # Different expert paths share a seed but do not share instantaneous poses.
                state_error=max(abs(x-y) for x,y in zip(left[index]['state'],right[index]['state']))
                entry=dict(index=index,state_error_rad=state_error,pixel_comparable=a.replay)
                if a.replay:entry.update(image_metrics(*images))
                entries.append(entry)
                panel=np.concatenate(images,axis=1)
                cv2.putText(panel,f'{camera} frame {index}; reference | candidate',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                sheets.append(panel)
        finally:
            for capture in captures:capture.release()
        metrics[camera]=entries
    cv2.imwrite(str(a.output/'side-by-side.jpg'),np.concatenate(sheets,axis=0))
    profiles=[]
    receipt=a.candidate/'receipt.json'
    if receipt.exists():profiles=json.loads(receipt.read_text()).get('profiles',[])
    report=dict(reference_frames=len(left),candidate_frames=len(right),images=metrics,stages=stage_summary(profiles),
        comparison='same physical rows, renderer gap' if a.replay else 'same-seed trajectory comparison; pixels are not pose-matched')
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
