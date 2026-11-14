"""Standalone collection; launch only through launch.py after a lead window grant."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import shared
from storage import confined,atomic_json,wait_for_storage
from collection_status import begin,finish,terminal
from contract import parameters,JOINTS,timing_audit
from episode_policy import QualityReject,FailureStreak,disposition
from success import initial_arm,settled_containment
from recovery import receipt_error
from tick_contract import digest,require_snapshots


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--fixture',type=Path,action='append',required=True,
        help='ROS receipt containing params and initial_physical; repeat for matched-seed layouts')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',required=True,help='Lead-granted window identity (recorded)')
    p.add_argument('--kind',choices=['normal','regrasp','both'],default='both')
    p.add_argument('--per-kind',type=int,default=5)
    p.add_argument('--max-attempts',type=int,default=30)
    p.add_argument('--physics-only',action='store_true')
    p.add_argument('--step-mode',choices=['held','playing'],default='held')
    p.add_argument('--instance',type=int,default=31)
    return p.parse_args(argv)


def fixture_pools(fixtures,kinds,per_kind):
    pools={kind:[f for f in fixtures if f['params'].get('recovery','none')==
        ('regrasp' if kind=='regrasp' else 'none')] for kind in kinds}
    for kind,pool in pools.items():
        if len({f['params']['seed'] for f in pool})<per_kind:
            raise ValueError('At least per-kind distinct '+kind+' fixture seeds are required for matched ROS comparisons')
    return pools


def next_fixture(pool,attempt,accepted_seeds):
    remaining=[fixture for fixture in pool if fixture['params']['seed'] not in accepted_seeds]
    if not remaining:raise ValueError('No unaccepted fixture seed remains for this recovery kind')
    return remaining[attempt%len(remaining)]


def run(a):
    import threading
    import numpy as np
    root=confined(a.output);root.mkdir(parents=True,exist_ok=False)
    certificate=begin(root);app=None;backend=None;cleanup=[];error=None
    stopped=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *args:stopped.set())
    start=time.perf_counter();counts=dict(normal=0,regrasp=0);receipts=[]
    try:
        from watch import heartbeat
        heartbeat('startup',900.)
        if a.per_kind<1 or a.max_attempts<1:raise ValueError('Positive episode and attempt limits required')
        if a.physics_only and a.step_mode=='playing':raise ValueError('Playing mode requires its camera producers')
        fixtures=[]
        for path in a.fixture:
            value=json.loads(path.read_text())
            if not {'params','initial_physical'}<=value.keys():raise ValueError('Fixture lacks original physical snapshot and seed')
            fixtures.append(value)
        kinds=['normal','regrasp'] if a.kind=='both' else [a.kind]
        pools=fixture_pools(fixtures,kinds,a.per_kind);kind_attempts={k:0 for k in kinds}
        accepted_seeds={k:set() for k in kinds}
        from isaacsim import SimulationApp
        app=SimulationApp(shared.app_config())
        from native import Native
        from safety import Geometry
        from core import Capture
        from expert import Expert
        from writer import Writer
        from replay_contract import packet
        from so_arm101_control.grasp_geometry import _grasp_home_data_target
        heartbeat('native initialization',900.)
        if a.step_mode=='playing':
            from playing import Playing
            backend=Playing(a.scene,a.manifest)
        else:backend=Native(a.scene,a.manifest,cameras=not a.physics_only)
        # Bind CUDA, renderer settings and current Kit table independently.
        import carb
        import torch
        from gpu_device import inventory,audit_sample
        from nvml_processes import sample
        from renderer_device import read_table,resolve
        from run_smoke import identity
        if torch.cuda.device_count()!=1:raise RuntimeError('Exactly one CUDA-visible GPU required')
        settings=carb.settings.get_settings()
        if settings.get('/renderer/multiGpu/enabled') is not False:raise RuntimeError('Multi-GPU is enabled')
        def gpu_check():
            cards=inventory();assigned=os.environ['SOARM_ASSIGNED_GPU_UUID']
            own=identity(os.getpid());observed=sample(cards)
            result=audit_sample(assigned,{own['pid']:own},cards,observed)
            rows,source=read_table(Path(os.environ['SOARM_CURRENT_KIT_LOG']))
            result['renderer']=dict(resolve(rows,cards,assigned,require_active=True),source=source)
            if not result['passed']:raise RuntimeError('GPU attribution failed: '+repr(result))
            return result
        gpu_start=gpu_check()
        heartbeat('capture/expert',120.)
        geometry=Geometry(backend.manifest['geometry']);capture=Capture(backend,geometry)
        source_files=[*Path(__file__).parent.glob('*.py'),
            * (shared.REPO/'linux-env/scripts/collect').glob('*.py'),
            shared.REPO/'isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json']
        config=dict(version=1,instance=a.instance,fps=30,backend='inprocess',physics='cpu',
            native_pipeline_settings=getattr(backend,'pipeline_settings',{}),
            physics_only=a.physics_only,step_mode=a.step_mode,window=a.window,normal_target=a.per_kind,regrasp_target=a.per_kind,
            scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),gpu_start=gpu_start,
            fixture_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.fixture},
            source_sha256={str(p.relative_to(shared.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            git_commit=subprocess.check_output(['git','-C',str(shared.REPO),'rev-parse','HEAD'],text=True).strip())
        atomic_json(root/'run.json',config)
        streak=FailureStreak()
        for trial in range(a.max_attempts):
            if all(counts[k]>=a.per_kind for k in kinds):break
            if stopped.is_set():raise RuntimeError('Collection interrupted')
            wait_for_storage(root,stopped)
            kind=next(k for k in kinds if counts[k]<a.per_kind)
            fixture=next_fixture(pools[kind],kind_attempts[kind],accepted_seeds[kind]);kind_attempts[kind]+=1
            p=parameters(fixture['params']['seed'],fixture['params']['color'],
                split=fixture['params'].get('split','train'),regrasp=kind=='regrasp')
            np.random.seed(p['seed']%2**32)
            episode_id=digest(dict(root=str(root),trial=trial,seed=p['seed'],kind=kind))
            path=root/f'attempt-{trial:04d}';writer=None;record=False;profiles=[];checked=[]
            receipt=dict(params=p,episode_id=episode_id,accepted=False,initial_physical=fixture['initial_physical'])
            def tick(reference,executed,offset,phase):
                if stopped.is_set():raise RuntimeError('Collection interrupted')
                heartbeat()
                s=backend.snapshot();images=None;render_profile={}
                transforms,cameras=backend.transforms(s)
                if a.step_mode=='playing':
                    row,snapshot,timing,images,render_profile,physics_ms=capture.playing_tick(
                        reference,executed,offset,phase,episode_id,s)
                else:
                    if not a.physics_only:images,render_profile=backend.render()
                    row,snapshot,timing=capture.tick(reference,executed,offset,phase,episode_id,images,s,
                        camera_stamps=render_profile.get('camera_stamps'))
                pose=packet(s,transforms,cameras,len(checked))
                pose['row_sha256']=digest(row)
                profile=dict(timing,**render_profile,sim_ns=row['sim_ns'])
                checked.append(snapshot)
                if record:profile['enqueue_ms']=writer.add(row,images,pose,profile)
                profile['physics_render_ms' if a.step_mode=='playing' else 'physics_ms']=(
                    physics_ms if a.step_mode=='playing' else backend.step())
                profiles.append(profile)
                return row,snapshot
            try:
                geometry.carried=None;geometry.contact=False;geometry.selected=None
                restored=backend.restore(fixture['initial_physical']);capture.boundary(restored)
                backend.step()
                home=_grasp_home_data_target();reference=[home[n] for n in JOINTS[:5]]+[backend.snapshot()['state'][5]]
                for _ in range(6):tick(reference,reference,[0.]*6,'home_verification')
                gate=initial_arm(checked[-6:],reference[:5]);receipt['initial_arm_check']=gate
                if not gate['contract_passed']:raise RuntimeError('Home tick contract failed')
                if not gate['passed']:raise QualityReject('home_unstable',gate,gate['contract'])
                # Zero velocities and establish a verified fresh capture boundary.
                restored=backend.restore(backend.snapshot());capture.boundary(restored)
                backend.step();checked.clear()
                writer=Writer(path,render=not a.physics_only);record=True
                expert=Expert(backend,geometry,tick,p,episode_id)
                result=expert.cycle();record=False
                receipt.update(script_completed=True,noise_operations=expert.noise_specs)
                if result['recovery_report']:receipt['recovery_report']=result['recovery_report']
                for _ in range(15):tick(expert.reference,expert.reference,[0.]*6,'containment_verification')
                score=settled_containment(checked[-15:],p['object'],p['color'])
                score['lift_m']=result['lift_m'];receipt['score']=score
                if not score['contract_passed']:raise RuntimeError('Containment tick contract failed')
                if not score['passed']:raise QualityReject('containment',score,score['contract'])
                receipt['accepted']=not a.physics_only;receipt['physics_candidate']=True
            except QualityReject as exc:
                receipt['quality_reject']=dict(kind=exc.kind,evidence=exc.evidence,contract=exc.contract)
            except BaseException as exc:
                receipt['error']=repr(exc);error=exc
            finally:
                record=False
                if writer:
                    try:writer.close()
                    except BaseException as exc:receipt['writer_error']=repr(exc);error=exc
                    receipt['timing']=timing_audit(writer.rows,30)
                    receipt['accepted']=receipt['accepted'] and receipt['timing']['passed']
                else:path.mkdir(parents=True,exist_ok=True)
                held=False;pause_failure=None
                try:
                    backend.world.pause();held=not backend.world.is_playing()
                    if not held:raise RuntimeError('World pause did not hold the simulation')
                except BaseException as exc:
                    pause_failure=repr(exc);receipt['pause_error']=pause_failure
                    if error is None:error=exc
                backend.epoch+=1
                receipt.update(end_cancel=dict(held=held,failure=pause_failure,epoch=backend.epoch),
                    retirement=dict(passed=writer is None or not writer.thread.is_alive(),workers=0),profiles=profiles)
                if receipt['accepted'] or receipt.get('physics_candidate'):
                    problem=receipt_error(receipt,writer.rows)
                    if problem:error=RuntimeError(problem)
                decision=disposition(receipt,fatal=error,rows=writer.rows if writer else [])
                if a.physics_only and error is None and receipt.get('physics_candidate') and receipt.get('timing',{}).get('passed'):
                    decision=dict(kind='accepted',reason='physics_candidate_only')
                if decision['kind']!='accepted':receipt['accepted']=False
                receipt['disposition']=decision
                atomic_json(path/'receipt.json',receipt)
                with (root/'episodes.jsonl').open('a') as out:out.write(json.dumps(receipt)+'\n')
                receipts.append(receipt)
                if decision['kind']=='accepted':
                    counts[kind]+=1;accepted_seeds[kind].add(p['seed'])
                stop=streak.observe(decision)
                if stop['stop']:
                    if error is None:error=RuntimeError('Collection stopped: '+repr(stop))
                    raise error
        if not all(counts[k]>=a.per_kind for k in kinds):raise RuntimeError('Episode quota not reached')
        atomic_json(root/'gpu-final.json',gpu_check())
        summary=dict(complete=not a.physics_only,counts=counts,attempts=len(receipts),
            accepted_seeds={k:sorted(seeds) for k,seeds in accepted_seeds.items()},
            producer_invocation=certificate['invocation'],wall_s=time.perf_counter()-start,
            end_marker='PHYSICS_CAPTURE_FINISHED' if a.physics_only else 'COLLECTION_FINISHED')
        atomic_json(root/'summary.json',summary)
    except BaseException as exc:
        error=exc;terminal(root,dict(source='inprocess',error=repr(exc)))
    finally:
        try:
            from watch import heartbeat
            heartbeat('shutdown',180.)
        except BaseException as exc:cleanup.append(dict(source='heartbeat',error=repr(exc)))
        for label,obj in (('backend',backend),):
            if obj:
                try:obj.close()
                except BaseException as exc:cleanup.append(dict(source=label,error=repr(exc)))
        if cleanup:terminal(root,dict(source='cleanup',errors=cleanup))
        shared.publish_exit_ready(root,certificate,error is None and not cleanup)
        if app:app.close(skip_cleanup=True,exit_code=0 if error is None and not cleanup else 1)
        finish(root,certificate,cleanup)
    if error:raise error
    print('PHYSICS_CAPTURE_FINISHED' if a.physics_only else 'COLLECTION_FINISHED',flush=True)


if __name__=='__main__':run(arguments())
