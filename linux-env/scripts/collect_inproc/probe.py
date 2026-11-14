"""Owned standalone boot/cadence/RGB probe; no ROS episode fixtures or collection."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import hashlib
import shared
from storage import confined,atomic_json
from collection_status import begin,finish,terminal


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window',required=True)
    p.add_argument('--instance',type=int,default=2);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--step-mode',choices=['held','playing'],default='held')
    args=p.parse_args(argv)
    if args.frames<1:p.error('--frames must be positive')
    return args


def main(a):
    from watch import heartbeat
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False)
    cert=begin(root);app=None;backend=None;cleanup=[];failure=None
    report=dict(window=a.window,step_mode=a.step_mode,episodes_collected=0,checks={},frames=[])
    start=time.perf_counter()
    def save(stage):
        report['stage']=stage;report['wall_s']=time.perf_counter()-start
        atomic_json(root/'probe.json',report)
    try:
        heartbeat('probe startup',900.);save('app_start')
        from isaacsim import SimulationApp
        app=SimulationApp(shared.app_config())
        from native import Native
        from safety import Geometry
        from core import Capture
        from gpu_device import inventory,audit_sample
        from renderer_device import read_table,resolve
        from nvml_processes import sample
        from run_smoke import identity
        import torch
        import carb
        import numpy as np
        from pxr import PhysxSchema
        heartbeat('probe',120.);save('native_init')
        if a.step_mode=='playing':
            from playing import Playing
            backend=Playing(a.scene,a.manifest)
        else:backend=Native(a.scene,a.manifest)
        own=identity(os.getpid());cards=inventory();assigned=os.environ['SOARM_ASSIGNED_GPU_UUID']
        gpu=audit_sample(assigned,{own['pid']:own},cards,sample(cards))
        rows,source=read_table(Path(os.environ['SOARM_CURRENT_KIT_LOG']))
        gpu['renderer']=dict(resolve(rows,cards,assigned,require_active=True),source=source)
        report['gpu']=gpu
        report['checks']['gpu']=gpu['passed'] and torch.cuda.device_count()==1 and os.environ['CUDA_VISIBLE_DEVICES']==assigned
        report['nice']=os.getpriority(os.PRIO_PROCESS,0)
        report['domain']=os.environ.get('ROS_DOMAIN_ID')
        report['checks']['nice_domain']=report['nice']==10 and report['domain']==str(90+a.instance)
        scene=backend.stage.GetPrimAtPath(backend.world.get_physics_context().prim_path)
        report['checks']['cpu_physx']=PhysxSchema.PhysxSceneAPI(scene).GetEnableGPUDynamicsAttr().Get() is False
        geometry=Geometry(backend.manifest['geometry'])
        geometry.update(backend.snapshot());report['checks']['native_fk']=True
        restored=backend.restore(backend.manifest['baseline'])
        report['restore']=restored;report['checks']['restore']=True
        capture=Capture(backend,geometry);capture.boundary(restored)
        save('capture')
        # Verify a second epoch after the uninterrupted capture sequence.
        for index in range(a.frames+2):
            heartbeat()
            if index==a.frames:
                restored=backend.restore(backend.manifest['baseline'])
                capture.boundary(restored)
                report['second_restore']=restored
            counts=dict(backend.rgb.render_events)
            old_index=backend.world.current_time_step_index;old_time=backend.world.current_time
            physics_ms=backend.step();s=backend.snapshot()
            report['last_physical_advance']=dict(steps=backend.world.current_time_step_index-old_index,
                seconds=backend.world.current_time-old_time,sim_ns=s['sim_ns'])
            save('capture')
            images,timing=backend.render()
            row,checked,contract_timing=capture.tick(s['state'],s['state'],[0.]*6,
                'boot_probe','boot_probe',images,s,camera_stamps=timing.get('camera_stamps'))
            counts_after=dict(backend.rgb.render_events)
            delta={k:counts_after.get(k,0)-counts.get(k,0) for k in set(counts)|set(counts_after)}
            report['frames'].append(dict(index=index,epoch=s['epoch'],sim_ns=s['sim_ns'],**{
                'physics_render_ms' if a.step_mode=='playing' else 'physics_ms':physics_ms},
                physical_steps=backend.world.current_time_step_index-old_index,
                physical_seconds=backend.world.current_time-old_time,render_completions=delta,
                consumed_pairs=backend.rgb.calls,images=row['camera_identity'],**timing,**contract_timing))
            if index in (0,a.frames-1,a.frames+1):
                from PIL import Image
                for key,img in images.items():Image.fromarray(img).save(root/(key+'-'+str(index)+'.png'))
            save('capture')
        before=backend.snapshot();counts=dict(backend.rgb.render_events)
        if a.step_mode=='playing':
            frame_ids={k:p.hydra_texture.get_frame_info().get('swh_frame_number') for k,p in backend.rgb.products.items()}
            for _ in range(1000):hashlib.sha256(b'inprocess expert work between app updates').digest()
            same_ids=frame_ids=={k:p.hydra_texture.get_frame_info().get('swh_frame_number') for k,p in backend.rgb.products.items()}
        else:
            same_ids=True
            settings=carb.settings.get_settings();old=settings.get('/app/player/playSimulations')
            settings.set('/app/player/playSimulations',False)
            try:
                for _ in range(12):app.update()
            finally:settings.set('/app/player/playSimulations',old)
        after=backend.snapshot();counts_after=dict(backend.rgb.render_events)
        report['held_updates']=dict(updates=0 if a.step_mode=='playing' else 12,unchanged=before==after,
            native_frame_ids_unchanged=same_ids,
            render_completions={k:counts_after.get(k,0)-counts.get(k,0) for k in set(counts)|set(counts_after)})
        report['render_event_samples']=backend.rgb.event_samples
        report['checks']['cadence']=all(f['physical_steps']==4 for f in report['frames'])
        report['checks']['rgb']=all(f['consumed_pairs']==1 for f in report['frames'])
        report['checks']['held_no_render']=before==after and same_ids and not any(report['held_updates']['render_completions'].values())
        report['checks']['tick_contract']=True
        if not all(report['checks'].values()):raise RuntimeError('Boot probe check failed: '+repr(report['checks']))
        save('verified')
    except BaseException as exc:
        failure=exc;report['error']=repr(exc);report['traceback']=traceback.format_exc()
        if backend and backend.rgb:
            report['render_event_samples']=backend.rgb.event_samples
            report['render_completions_total']=dict(backend.rgb.render_events)
            report['failed_capture_callbacks']=backend.rgb.calls
        terminal(root,dict(source='boot_probe',error=repr(exc)));save('failed')
    finally:
        heartbeat('shutdown',180.)
        for name,obj in (('backend',backend),):
            if obj:
                try:obj.close()
                except BaseException as exc:cleanup.append(dict(source=name,error=repr(exc)))
        if cleanup:terminal(root,dict(source='probe_cleanup',errors=cleanup))
        report['cleanup_errors']=cleanup;report['passed']=failure is None and not cleanup
        save('complete' if report['passed'] else 'failed')
        if report['passed']:
            atomic_json(root/'summary.json',dict(complete=True,end_marker='BOOT_PROBE_FINISHED',
                producer_invocation=cert['invocation'],kind='boot_probe',episodes=0))
        # The external owner finalizes only after the Kit process and all its
        # descendants have exited; fast shutdown does not return to Python.
        shared.publish_exit_ready(root,cert,report['passed'])
        if app:app.close(skip_cleanup=True,exit_code=0 if report['passed'] else 1)
        finish(root,cert,cleanup)
    if failure:raise failure
    if cleanup:raise RuntimeError('Probe cleanup failed')
    print('BOOT_PROBE_FINISHED',flush=True)


if __name__=='__main__':main(arguments())
