"""Diagnostic media from an attempted episode; no additional simulation ticks."""
import subprocess


def final_stills(path, rows, run=subprocess.run):
    if not rows:
        return dict(saved=False,reason='Attempt has no recorded frames')
    last=len(rows)-1
    files={}
    for key in ('wrist','top'):
        target=path/('final-'+key+'.png')
        run(['ffmpeg','-nostdin','-v','error','-y','-threads','1',
             '-i',str(path/(key+'.mp4')),'-vf',f'select=eq(n\\,{last})',
             '-frames:v','1','-threads','1',str(target)],check=True,timeout=20)
        files[key]=target.name
    return dict(saved=True,frame_index=last,sim_ns=rows[-1]['sim_ns'],files=files,
                meaning='Final recorded camera frame; settled verification tail is outside the training video')
