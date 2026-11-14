"""Owned standalone boot/cadence/RGB probe; no ROS episode fixtures or collection."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import hashlib
import shared
import numpy as np
from storage import confined,atomic_json
from collection_status import begin,finish,terminal


def view_pose_errors(view,before,after):
    """Compare native drawable view metadata to both measured camera poses."""
    matrix=np.asarray(view,float).reshape(4,4)
    if not np.isfinite(matrix).all():raise ValueError('Non-finite drawable camera view')
    expected=np.linalg.inv(np.asarray(before,float))
    choices=[matrix,matrix.T]
    errors=[float(np.max(np.abs(choice-expected))) for choice in choices]
    choice=int(np.argmin(errors))
    return dict(before_error=errors[choice],
        after_error=float(np.max(np.abs(choices[choice]-np.linalg.inv(np.asarray(after,float))))),
        view_layout='matrix' if choice==0 else 'transpose')


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window',required=True)
    p.add_argument('--instance',type=int,default=2);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--step-mode',choices=['held','playing'],default='held')
    p.add_argument('--motion-check',action='store_true',help='Collision-checked pan motion and camera-pose phase diagnostic')
    args=p.parse_args(argv)
    if args.frames<1:p.error('--frames must be positive')
    if args.motion_check and (args.step_mode!='playing' or args.frames<8):
        p.error('--motion-check needs playing mode and at least eight frames')
    return args


def main(a):
    from watch import heartbeat
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False)
    cert=begin(root);app=None;backend=None;cleanup=[];failure=None
    report=dict(window=a.window,step_mode=a.step_mode,motion_check=a.motion_check,episodes_collected=0,checks={},frames=[])
    start=time.perf_counter()
    def save(stage):
        report['stage']=stage;report['wall_s']=time.perf_counter()-start
        if backend:report['priming_reports']=getattr(backend,'priming_reports',[])
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
        heartbeat('native initialization',900.);save('native_init')
        if a.step_mode=='playing':
            from playing import Playing
            backend=Playing(a.scene,a.manifest)
        else:backend=Native(a.scene,a.manifest)
        heartbeat('probe',120.)
        report['warmup_rgb']=getattr(backend,'warmup_rgb',None)
        report['warmup_capture_settings']=getattr(backend,'warmup_capture_settings',None)
        report['pipeline_settings']=getattr(backend,'pipeline_settings',None)
        report['warmup_graph']=getattr(backend,'warmup_graph',None)
        if a.step_mode=='playing':
            from PIL import Image
            for key,annotation in backend.rgb.annotators.items():
                pixels=np.asarray(annotation.get_data())
                if pixels.dtype==np.uint8 and pixels.shape==(480,640,4):
                    Image.fromarray(pixels[:,:,:3]).save(root/('diagnostic-warmup-'+key+'.png'))
        report['cache_settings']={key:carb.settings.get_settings().get('/rtx/shaderDb/'+key)
            for key in ('shaderCachePath','driverShaderCachePath')}
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
        if a.step_mode=='playing':backend.step()
        save('capture')
        # Verify a second epoch after the uninterrupted capture sequence.
        for index in range(a.frames+2):
            heartbeat()
            if index==a.frames:
                if a.step_mode=='playing':backend.world.pause()
                restored=backend.restore(backend.manifest['baseline'])
                capture.boundary(restored)
                if a.step_mode=='playing':backend.step()
                report['second_restore']=restored
            counts=dict(backend.rgb.render_events)
            old_index=backend.world.current_time_step_index;old_time=backend.world.current_time
            if a.step_mode=='playing':
                s=backend.snapshot()
                reference=list(s['state'])
                if a.motion_check and index<a.frames:
                    reference=list(backend.manifest['baseline']['state'])
                    if index>=a.frames//2:reference[0]+=.12
                from contract import corrective_action
                execution=corrective_action(s['state'],reference)
                row,checked,contract_timing,images,timing,physics_ms=capture.playing_tick(
                    reference,execution,[0.]*6,'boot_probe','boot_probe',s)
            else:
                physics_ms=backend.step();s=backend.snapshot()
            report['last_physical_advance']=dict(steps=backend.world.current_time_step_index-old_index,
                seconds=backend.world.current_time-old_time,sim_ns=backend.stamp(),observation_ns=s['sim_ns'])
            save('capture')
            if a.step_mode!='playing':
                images,timing=backend.render()
                row,checked,contract_timing=capture.tick(s['state'],s['state'],[0.]*6,
                    'boot_probe','boot_probe',images,s,camera_stamps=timing.get('camera_stamps'))
            counts_after=dict(backend.rgb.render_events)
            delta={k:counts_after.get(k,0)-counts.get(k,0) for k in set(counts)|set(counts_after)}
            pose_phase={}
            if a.motion_check:
                after=backend.snapshot()
                _,camera_before=backend.transforms(s);_,camera_after=backend.transforms(after)
                pose_phase=dict(observation_state=s['state'],arrival_state=after['state'],
                    action=row['action'],executed_action=row['executed_action'],
                    camera_world_before=camera_before,camera_world_after=camera_after,
                    camera_pose_phase={key:view_pose_errors(timing['playing_frame']['cameras'][key]['render_view'],
                        camera_before[key],camera_after[key]) for key in ('wrist','top')})
            report['frames'].append(dict(index=index,epoch=s['epoch'],sim_ns=s['sim_ns'],**{
                'physics_render_ms' if a.step_mode=='playing' else 'physics_ms':physics_ms},
                physical_steps=backend.world.current_time_step_index-old_index,
                physical_seconds=backend.world.current_time-old_time,render_completions=delta,
                consumed_pairs=backend.rgb.calls,images=row['camera_identity'],**{**timing,**contract_timing,**pose_phase}))
            if index in (0,a.frames//2,a.frames//2+1,a.frames-1,a.frames+1):
                from PIL import Image
                for key,img in images.items():Image.fromarray(img).save(root/(key+'-'+str(index)+'.png'))
            save('capture')
        before=backend.snapshot();counts=dict(backend.rgb.render_events)
        if a.step_mode=='playing':
            frame_ids={k:v.get('frame_number') for k,v in backend.rgb.frame_info.items()}
            for _ in range(1000):hashlib.sha256(b'inprocess expert work between app updates').digest()
            same_ids=frame_ids=={k:v.get('frame_number') for k,v in backend.rgb.frame_info.items()}
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
        if a.motion_check:
            moving=report['frames'][:a.frames]
            pan0=backend.manifest['baseline']['state'][0]
            report['checks']['motion_measured']=max(abs(f['observation_state'][0]-pan0) for f in moving)>.01
            phases=[f['camera_pose_phase']['wrist'] for f in report['frames']]
            report['checks']['camera_pose_phase']=(all(p['before_error']<2e-4 for p in phases)
                and any(p['after_error']-p['before_error']>1e-4 for p in phases))
        if not all(report['checks'].values()):raise RuntimeError('Boot probe check failed: '+repr(report['checks']))
        save('verified')
    except BaseException as exc:
        failure=exc;report['error']=repr(exc);report['traceback']=traceback.format_exc()
        if backend and backend.rgb:
            report['failed_playing_frame']=getattr(backend,'last_frame',None)
            report['render_event_samples']=backend.rgb.event_samples
            report['render_completions_total']=dict(backend.rgb.render_events)
            report['failed_capture_callbacks']=backend.rgb.calls
        terminal(root,dict(source='boot_probe',error=repr(exc)));save('failed')
        if (a.step_mode=='playing' and backend and all(report['checks'].values()) and
            str(exc).startswith('Expected exactly one new render')):
            report['unaccepted_capture_trace']=backend.trace_steps()
            save('failed')
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
