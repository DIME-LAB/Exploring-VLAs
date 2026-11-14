"""Render immutable physics-pass pose packets in a separate, physics-free Kit.

Produces replay videos and a completion receipt. It does not promote physical
candidates to accepted datasets; live/replay fidelity review is a separate gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import shared
from storage import confined,atomic_json
from replay_contract import verify
from tick_contract import digest


def apply_poses(stage,poses,seconds):
    from pxr import UsdGeom,Usd,Gf
    from native import usd_matrix
    stage.SetTimeCodesPerSecond(30.)
    code=Usd.TimeCode(seconds*30.)
    # Author every world pose directly, ignoring parent transforms but retaining
    # each prim's complete original matrix scale in the recorded rigid transform.
    for path,matrix in poses.items():
        prim=stage.GetPrimAtPath(path)
        if not prim.IsValid():raise RuntimeError('Replay prim missing: '+path)
        x=UsdGeom.Xformable(prim)
        # Matrices contain rigid transforms; preserve original prim scale.
        local=np.array(x.GetLocalTransformation()).T
        scale=np.linalg.norm(local[:3,:3],axis=0)
        target=np.asarray(matrix,float).copy();target[:3,:3]*=scale
        x.ClearXformOpOrder();x.SetResetXformStack(True)
        op=x.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,'inprocReplay')
        op.Set(Gf.Matrix4d(target.T.tolist()))
        op.Set(Gf.Matrix4d(target.T.tolist()),code)
        actual=usd_matrix(prim)
        if not np.allclose(actual,target,atol=1e-6,rtol=0):raise RuntimeError('Replay pose readback differs: '+path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',required=True)
    a=p.parse_args();output=confined(a.output);output.mkdir(parents=True,exist_ok=False)
    app=None;writer=None;failure=None;profiles=[]
    start=time.perf_counter()
    try:
        manifest=json.loads(a.manifest.read_text())
        if hashlib.sha256(a.scene.read_bytes()).hexdigest()!=manifest['scene_sha256']:raise RuntimeError('Replay scene differs')
        rows=[json.loads(line) for line in (a.attempt/'frames.jsonl').read_text().splitlines()]
        packets=[json.loads(line) for line in (a.attempt/'poses.jsonl').read_text().splitlines()]
        if not rows or len(rows)!=len(packets):raise RuntimeError('Replay rows/poses are incomplete')
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,hide_ui=True,active_gpu=int(os.environ['SOARM_RENDERER_GPU_INDEX']),
            physics_gpu=0,multi_gpu=False,disable_viewport_updates=True,fast_shutdown=False))
        import omni.usd
        import omni.timeline
        from pxr import UsdPhysics,UsdGeom
        from writer import Writer
        from rgb_capture import RGB
        omni.usd.get_context().open_stage(str(a.scene))
        stage=omni.usd.get_context().get_stage()
        for root in ('/Graph','/Render/PostProcess'):stage.RemovePrim(root)
        timeline=omni.timeline.get_timeline_interface();timeline.stop();timeline.set_auto_update(False)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
            if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr().Set(False)
        rgb=RGB(manifest['camera_paths'])
        writer=Writer(output/'rendered',render=True)
        previous=None;paths=set(manifest['body_paths'].values())|set(manifest['link_paths'].values())
        for index,(row,record) in enumerate(zip(rows,packets)):
            frame_start=time.perf_counter_ns();packet=verify(record,previous,paths)
            if record['row_sha256']!=digest(row) or any(packet[a]!=row[b] for a,b in (
                ('sim_ns','pose_stamp_ns'),('epoch','epoch'),('state','physics_state'),
                ('bodies','physical_bodies'),('physics_time_s','physics_time_s'))):
                raise RuntimeError('Replay state is not bound to the raw physics row')
            poses=dict(packet['transforms'])
            poses.update({manifest['camera_paths'][k]:v for k,v in packet['cameras'].items()})
            seconds=packet['physics_time_s']
            apply_poses(stage,poses,seconds);timeline.set_current_time(seconds)
            images,timing=rgb.read(seconds,app.update)
            profile=dict(index=index,render_ms=(time.perf_counter_ns()-frame_start)/1e6,**timing)
            writer.add(row,images,record,profile);profiles.append(profile);previous=record
        writer.close();writer=None
        rgb.close()
    except BaseException as exc:failure=repr(exc)
    finally:
        if writer:
            try:writer.close()
            except BaseException as exc:failure=repr(exc)
        if app:
            try:app.close()
            except BaseException as exc:failure=repr(exc)
        atomic_json(output/'replay-receipt.json',dict(passed=failure is None,error=failure,
            window=a.window,frames=len(profiles),wall_s=time.perf_counter()-start,profiles=profiles,
            accepted_for_training=False,reason='live/replay fidelity gate pending'))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
