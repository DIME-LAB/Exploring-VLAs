"""Unscored live qualification of normal collector preparation and native rejoin."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from native_replay import REPO,require_environment,validate_collector_cut
from held_tick import CollectorTransport
from native_bank import load_bank
from matched_reset_client import reset_scene
from collect import mcp
from native_environment import qualified_host

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--bank-sha256',required=True);p.add_argument('--collector-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require_environment();sources=validate_collector_cut();rows,suite=load_bank(a.bank,a.bank_sha256);assert suite=='seen-training-home'
    host=qualified_host(REPO);assert a.out.resolve().is_relative_to(Path(host['root'])/'eval/runs') and not a.out.exists();a.out.mkdir()
    transport=CollectorTransport();results=[]
    try:
        mcp('run_python',{'code':(REPO/'linux-env/scripts/collect/configure_clock.py').read_text()+'\nresult=configure(30)'})
        installed=mcp('run_python',{'code':f'import sys\nsys.path.insert(0,{str(REPO/"linux-env/scripts/collect")!r})\nimport collection_runtime as c\nresult=c.install("collect")'});transport.generation.install(installed)
        deadline=time.monotonic()+20
        while transport.publisher.get_subscription_count()<1:
            if time.monotonic()>deadline:raise TimeoutError('Native reset command discovery')
            time.sleep(.05)
        mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read()
        for index,row in enumerate(rows):
            out=a.out/f'ep{index:02d}';out.mkdir();started=time.monotonic()
            # Capture the previous one-frame protocol as an unscored control.
            old=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().restore('+repr(dict(state=row['state'],drive_targets=row['recorded_drive_targets'],bodies=row['bodies']))+',arm=True,restore_drives=True)'})
            transport.generation.install(old);transport.sensors.prune_durable_camera_history();mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read()
            np.savez_compressed(out/'one-bootstrap-cameras.npz',**{cam:transport.sensors.image(transport.session.entry[cam][0]).copy() for cam in ('top','wrist')})
            reset,proof=reset_scene(transport,REPO,a.collector_root,row,out/'matched-reset');snapshot=transport.session.current
            images={cam:transport.sensors.image(transport.session.entry[cam][0]).copy() for cam in ('top','wrist')};assert all(im.shape==(480,640,3) and im.dtype==np.uint8 for im in images.values())
            np.savez_compressed(out/'matched-cameras.npz',**images)
            body_error=max(abs(x-y) for name in row['bodies'] for x,y in zip(snapshot['bodies'][name]['position'],row['bodies'][name]['position']))
            joint_error=float(np.max(abs(np.degrees(np.array(snapshot['state'])-row['state']))));assert body_error<=1e-4 and joint_error<=.01
            commands=[];snaps=[snapshot];hold=snapshot['state']
            for _ in range(10):
                command,current=transport.session.step(hold);commands.append(command);snaps.append(current)
            result=dict(status='PASS',seed=row['seed'],suite=suite,scored=False,reset_protocol='collector-matched-reset-v1',matched_reset=proof,joint_max_error_deg=joint_error,body_max_error_m=body_error,verified_hold_commands=len(commands),wall_s=time.monotonic()-started,first_snapshot=snapshot,cameras={k:dict(shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in images.items()})
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'commands.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in commands));(out/'physical.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in snaps));results.append(result);print(json.dumps(dict(seed=row['seed'],status='PASS',native_commands=len(commands))),flush=True)
        (a.out/'summary.json').write_text(json.dumps(dict(status='PASS',scored=False,episodes=len(results),commands=sum(r['verified_hold_commands'] for r in results),reset_collector_root=str(a.collector_root.resolve()),rows=results,collector_source_sha256=sources,bank_sha256=a.bank_sha256),indent=2)+'\n')
    finally:transport.close()
if __name__=='__main__':main()
