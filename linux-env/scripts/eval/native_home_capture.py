"""Capture both cameras at two sealed training home states; no policy."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from native_replay import require_environment,validate_collector_cut,REPO
from held_tick import CollectorTransport
from collect import mcp
from native_environment import qualified_host

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--bank-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require_environment();sources=validate_collector_cut();host=qualified_host(REPO)
    assert a.out.resolve().is_relative_to(Path(host['root'])/'eval/runs') and not a.out.exists();a.out.mkdir()
    manifest=json.loads((a.bank/'bank-manifest.json').read_text());assert hashlib.sha256((a.bank/'bank-manifest.json').read_bytes()).hexdigest()==a.bank_sha256
    for name,sha in manifest['files_sha256'].items():assert hashlib.sha256((a.bank/name).read_bytes()).hexdigest()==sha
    transport=CollectorTransport();rows=[]
    try:
        deadline=time.monotonic()+20
        while transport.publisher.get_subscription_count()<1:
            if time.monotonic()>deadline:raise TimeoutError('Native capture command graph discovery')
            time.sleep(.05)
        for seed in (530000,530002):
            row=json.loads((a.bank/f'state-{seed}.json').read_text());bodies=json.loads((a.bank/f'seed-{seed}.json').read_text())
            previous_epoch=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().epoch'})
            reset=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().restore('+repr(dict(state=row['state'],drive_targets=row['recorded_drive_targets'],bodies=bodies))+',arm=True,restore_drives=True)'})
            transport.generation.install(reset);transport.sensors.prune_durable_camera_history();mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read();snapshot=transport.session.current
            images={camera:transport.sensors.image(transport.session.entry[camera][0]).copy() for camera in ('top','wrist')}
            identity=mcp('run_python',{'code':'import collection_runtime as c\nr=c.runtime()\nresult=dict(mode=r.mode,held=not r.timeline.is_playing(),epoch=r.epoch,sim_ns=r.stamp(),busy=r.busy,failure=r.failure)'})
            assert identity['mode']=='collect' and identity['held'] and not identity['busy'] and identity['failure'] is None
            assert identity['epoch']==snapshot['epoch'] and identity['epoch']!=previous_epoch and abs(identity['sim_ns']-snapshot['sim_ns'])<=1000
            assert all(image.shape==(480,640,3) and image.dtype==np.uint8 for image in images.values())
            joint_error=float(np.max(abs(np.degrees(np.asarray(snapshot['state'])-row['state']))))
            body_error=max(abs(x-y) for name in bodies for x,y in zip(snapshot['bodies'][name]['position'],bodies[name]['position']))
            quaternion_error=max(min(max(abs(x-y) for x,y in zip(snapshot['bodies'][name]['orientation'],bodies[name]['orientation'])),max(abs(x+y) for x,y in zip(snapshot['bodies'][name]['orientation'],bodies[name]['orientation']))) for name in bodies)
            assert joint_error<=.01 and body_error<=1e-4 and quaternion_error<=1e-4
            np.savez_compressed(a.out/f'seed-{seed}-cameras.npz',**images)
            (a.out/f'seed-{seed}-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
            rows.append(dict(seed=seed,joint_max_error_deg=joint_error,body_max_position_error_m=body_error,body_max_quaternion_component_error=quaternion_error,identity=identity,epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],cameras={camera:dict(shape=list(image.shape),rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()) for camera,image in images.items()}))
        (a.out/'result.json').write_text(json.dumps(dict(status='PASS',scope='Two restored home layouts, both raw cameras; pixel parity requires visual/comparative inspection',bank_sha256=a.bank_sha256,source_sha256=sources,rows=rows),indent=2)+'\n')
        print(json.dumps(dict(status='PASS',captures=len(rows),camera_frames=2*len(rows))),flush=True)
    finally:transport.close()
if __name__=='__main__':main()
