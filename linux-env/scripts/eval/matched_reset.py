"""Opt-in scene-only worker using the collector's exact normal preparation path."""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,signal,subprocess,sys,time

COLLECTOR_SHA='09f86817b46e2b1ad4695fce29e059ea387a0bbc97d9568786a70c818308ae90'
HELPER_SHA='eb41cef2e7bee6a972e646a26963c8a0d01142895223969c880144401e4ee868'

def orientation_errors(actual,expected):
    cup={};lego={}
    for name,body in expected['bodies'].items():
        q=actual['bodies'][name]['orientation'];p=body['orientation']
        component=min(max(abs(x-y) for x,y in zip(q,p)),max(abs(x+y) for x,y in zip(q,p)))
        if name.startswith('cup_'):
            cosine=abs(sum(x*y for x,y in zip(q,p)))/math.sqrt(sum(x*x for x in q)*sum(x*x for x in p))
            cup[name]=math.degrees(2*math.acos(min(1.,max(0.,cosine))))
        else:lego[name]=component
    return dict(cup_geodesic_degrees=cup,lego_quaternion_component=lego,passed=max(cup.values())<=.1 and max(lego.values())<=1e-4)

def validate_collector_root(root):
    root=Path(root).resolve();folder=root/'linux-env/scripts/collect'
    assert hashlib.sha256((folder/'collect.py').read_bytes()).hexdigest()==COLLECTOR_SHA
    assert hashlib.sha256((folder/'scene_bank.py').read_bytes()).hexdigest()==HELPER_SHA
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()=='bf7336d93d5ba054c5edc66cea685725eb30eb40'
    subprocess.run(['git','-C',str(root),'diff','--exit-code','HEAD','--','linux-env/scripts/collect',':(exclude)linux-env/scripts/collect/collect.py','isaac/exts','linux-env/scripts/collect_sim.sh','linux-env/scripts/isaac'],check=True)
    return dict(root=str(root),collector_sha256=COLLECTOR_SHA,scene_helper_sha256=HELPER_SHA)

def main():
    p=argparse.ArgumentParser();p.add_argument('--collector-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.collector_root.resolve();folder=root/'linux-env/scripts/collect'
    source_validation=validate_collector_root(root)
    raw=a.receipt.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.receipt_sha256;source=json.loads(raw);reference=json.loads(a.reference_run.read_text())
    assert source['accepted'] and source['params']['recovery']=='none' and source['params']['seed'] in range(530000,530005)
    assert os.environ['ROS_DOMAIN_ID']=='92' and os.environ['SOARM_KIT_MCP_PORT']=='8781' and os.environ['RMW_IMPLEMENTATION']=='rmw_cyclonedds_cpp'
    assert all(k not in os.environ for k in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'))
    assert 'torch' not in sys.modules,'Scene preparation never imports a model or Torch'
    sys.path.insert(0,str(folder))
    import collect as c
    from storage import CANONICAL_ROOT,atomic_json,prepare_environment
    assert a.out.resolve().is_relative_to(CANONICAL_ROOT/'eval/runs') and not a.out.exists()
    prepare_environment(2);a.out.mkdir(parents=True)
    uri=os.environ['CYCLONEDDS_URI'];assert uri.startswith('file://') and hashlib.sha256(Path(uri[7:]).read_bytes()).hexdigest()=='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'
    args=argparse.Namespace(instance=2,output=a.out,resume=False,replay_receipt=None,normal_only=True,regrasp_only=False,episodes=1,max_attempts=1,seed=source['params']['seed'],episode_deadline=180.,wait_rtf_floor=.1,diagnostic=False)
    c.rclpy.init();collector=c.Collector(args);ok=False;started=time.monotonic()
    signal.signal(signal.SIGINT,lambda *_:collector.request_stop());signal.signal(signal.SIGTERM,lambda *_:collector.request_stop())
    try:
        with collector.status_lock:collector.finalization_certificate=c.collection_status.begin(a.out)
        collector.safe_home_baseline=None;collector.needs_home_restore=False
        clock,transport,live_baseline,sources,gpu,provenance=collector._initialize_scene_session(a.out,None,0)
        # Select the recorded training run's reset baseline. randomize_scene
        # itself restores CUP_LAYOUT before sampling, exactly as in one normal
        # multi-episode collector run; a prior policy episode is not a baseline.
        collector.baseline_cup_layout=copy.deepcopy(reference['baseline_cup_layout']);baseline=copy.deepcopy(reference['baseline'])
        atomic_json(a.out/'run.json',dict(protocol='collector-matched-reset-v1',baseline=baseline,baseline_cup_layout=collector.baseline_cup_layout,live_startup_baseline=live_baseline,collector_source_sha256=sources,source_reference_run=str(a.reference_run),source_reference_sha256=hashlib.sha256(a.reference_run.read_bytes()).hexdigest()))
        path=a.out/'attempt-0000';path.mkdir();receipt=dict(params=copy.deepcopy(source['params']),accepted=False,scene_only=True,episode_id=hashlib.sha256(str(a.out).encode()).hexdigest());epochs=set()
        collector._prepare_scene_attempt(receipt['params'],baseline,receipt,epochs,a.out,path)
        receipt['retirement']=collector.workers.retire();receipt['camera_boundary']=collector.sensors.require_camera_boundary(collector.generation.floor_ns);receipt['gpu_isolation_after']=collector.gpu_footprint()
        actual=collector.last_snapshot;expected=source['episode_start_physical']
        assert actual['sim_ns']==collector.episode_start_physical['sim_ns']
        with a.receipt.with_name('frames.jsonl').open() as stream:first_training=json.loads(stream.readline())
        training_offset=first_training['sim_ns']-expected['sim_ns'];assert abs(training_offset-1e9/30)<=1000
        import numpy as np
        q_error=float(np.max(abs(np.degrees(np.array(actual['state'])-expected['state']))))
        position_error=max(abs(x-y) for name in expected['bodies'] for x,y in zip(actual['bodies'][name]['position'],expected['bodies'][name]['position']))
        quaternion_error=max(min(max(abs(x-y) for x,y in zip(actual['bodies'][name]['orientation'],expected['bodies'][name]['orientation'])),max(abs(x+y) for x,y in zip(actual['bodies'][name]['orientation'],expected['bodies'][name]['orientation']))) for name in expected['bodies'])
        sampler_equal=receipt['cup_randomization']==source['cup_randomization']
        orientation=orientation_errors(actual,expected)
        validation=dict(passed=q_error<=.01 and position_error<=1e-4 and orientation['passed'] and sampler_equal,orientation_rule='Lead-approved cups geodesic<=0.1deg; LEGO quaternion component<=1e-4',orientation_errors=orientation,joint_max_error_deg=q_error,body_max_position_error_m=position_error,body_max_quaternion_error=quaternion_error,exact_cup_sampler=sampler_equal,joint_error_deg=np.degrees(np.array(actual['state'])-expected['state']).tolist(),body_position_errors_m={n:(np.array(actual['bodies'][n]['position'])-expected['bodies'][n]['position']).tolist() for n in expected['bodies']},actual=actual,expected=expected,actual_cup_sampler=receipt['cup_randomization'],expected_cup_sampler=source['cup_randomization'])
        atomic_json(a.out/'pose-validation.json',validation)
        receipt.update(prepared_snapshot=actual,pose_validation=validation)
        atomic_json(path/'receipt.json',receipt)
        print(json.dumps(dict(event='matched-reset-pose-validation',seed=source['params']['seed'],**{k:validation[k] for k in ('passed','joint_max_error_deg','body_max_position_error_m','body_max_quaternion_error','exact_cup_sampler')})),flush=True)
        assert validation['passed'],'Normal collector prepared pose differs from source; see pose-validation.json'
        assert receipt['retirement']['passed'] and receipt['camera_boundary']['passed'] and receipt['gpu_isolation_after']['passed']
        receipt.update(accepted=True,matched_reset_protocol='collector-matched-reset-v1',source_receipt=str(a.receipt),source_receipt_sha256=a.receipt_sha256,training_first_frame_offset_ns=training_offset,prepared_joint_max_error_deg=q_error,prepared_body_max_position_error_m=position_error,prepared_body_max_quaternion_error=quaternion_error,prepared_snapshot=actual,sequence='normal Collector._prepare_scene_attempt: reset_release,request_home,settle_home(before),scene(randomize6+barrier2),settle_home(after),restore_velocity,verification6,sync_scene',camera_priming='Exact normal collector functions; no fixed-N or render-only replacement',wall_s=time.monotonic()-started)
        atomic_json(path/'receipt.json',receipt);atomic_json(a.out/'summary.json',dict(producer_invocation=collector.finalization_certificate['invocation'],complete=True,scene_only=True,prepared=1,end_marker='COLLECTION_FINISHED'));ok=True
    except BaseException as error:
        atomic_json(a.out/'reset-error.json',dict(error=repr(error),scored=False))
        try:collector.close()
        except BaseException as cleanup_error:print('Secondary collector cleanup failure: '+repr(cleanup_error),file=sys.stderr,flush=True)
        raise
    else:
        collector.close()
    if ok:
        certificate=c.collection_status.require_finalized(a.out)
        assert 'torch' not in sys.modules,'Scene preparation imported Torch unexpectedly'
        atomic_json(a.out/'matched-reset-proof.json',dict(status='PASS',protocol='collector-matched-reset-v1',seed=source['params']['seed'],source_receipt_sha256=a.receipt_sha256,source_validation=source_validation,collector_sha256=COLLECTOR_SHA,scene_helper_sha256=HELPER_SHA,torch_imported=False,cyclonedds_uri=uri,cyclonedds_sha256=hashlib.sha256(Path(uri[7:]).read_bytes()).hexdigest(),finalization=certificate,receipt_sha256=hashlib.sha256((a.out/'attempt-0000/receipt.json').read_bytes()).hexdigest(),post_cleanup_semantics='Collector closes with native cancellation; caller must join a fresh epoch without re-teleporting'))
        print(json.dumps(dict(status='PASS',protocol='collector-matched-reset-v1',seed=source['params']['seed'])),flush=True)
if __name__=='__main__':main()
