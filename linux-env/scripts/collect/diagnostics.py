"""Diagnostic media from an attempted episode; no additional simulation ticks."""
import subprocess
import io


def png_dimensions(path):
    from PIL import Image
    raw=path.read_bytes()  # Missing files fail as well as zero-byte outputs.
    if not raw:
        raise ValueError(f'Empty diagnostic PNG: {path}')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG':
            raise ValueError(f'Diagnostic image is not PNG: {path}')
        image.load()
        return list(image.size)


def final_stills(path, rows, run=subprocess.run, verify=png_dimensions):
    if not rows:
        return dict(saved=False,reason='Attempt has no recorded frames')
    last=len(rows)-1
    files={};dimensions={}
    for key in ('wrist','top'):
        (path/('final-'+key+'.png')).unlink(missing_ok=True)
    for key in ('wrist','top'):
        target=path/('final-'+key+'.png')
        run(['ffmpeg','-nostdin','-v','error','-y','-threads','1',
             '-i',str(path/(key+'.mp4')),'-vf',f'select=eq(n\\,{last})',
             '-frames:v','1','-threads','1',str(target)],check=True,timeout=20)
        dimensions[key]=verify(target)
        files[key]=target.name
    return dict(saved=True,frame_index=last,sim_ns=rows[-1]['sim_ns'],files=files,dimensions=dimensions,
                meaning='Final recorded camera frame; settled verification tail is outside the training video')
