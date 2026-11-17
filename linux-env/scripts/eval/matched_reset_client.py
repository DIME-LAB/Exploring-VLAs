"""Invoke normal collector preparation before a native episode, without a model."""
from pathlib import Path
import hashlib,json,os,shlex,signal,subprocess,sys
from native_bank import SEEN_SUITES,SEEN15_SEEDS

def install_fenced_boundary(transport,native_mutation):
    """Use the collector's generation cancellation/retirement ordering."""
    retired={}
    def retire():
        with transport.sensors.condition:
            retired.update(pending_groups=len(transport.sensors.pending),acks=len(transport.sensors.acks))
            transport.sensors.pending.clear();transport.sensors.acks.clear()
            transport.session.current=None;transport.session.entry=None
    # Invalidate identity before retiring: in-flight camera callbacks finish
    # before the clear, and later callbacks reject the invalid/old floor.
    result=transport.generation.cancel(native_mutation,retire)
    transport.sensors.prune_durable_camera_history()
    return result,retired

def prepare_scene(repo,collector_root,row,out):
    repo=Path(repo);collector_root=Path(collector_root);out=Path(out)
    receipt=Path(row['source_receipt']);reference=Path(row['reference_run'])
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==row['source_receipt_sha256']
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==row['reference_run_sha256']
    env=dict(os.environ)
    text=subprocess.check_output(['python3',str(collector_root/'linux-env/scripts/collect/gpu_device.py'),'--gpu','0','--kit-log',str(Path(env['SOARM_INSTANCE_ROOT'])/'isaac.log')],text=True)
    for line in text.splitlines():
        parts=shlex.split(line);assert len(parts)==2 and parts[0]=='export';key,value=parts[1].split('=',1);env[key]=value
    assert env['CUDA_VISIBLE_DEVICES'].startswith('GPU-') and env['SOARM_RENDERER_GPU_INDEX']=='0'
    argv=['python3','-u',str(repo/'linux-env/scripts/eval/matched_reset.py'),'--collector-root',str(collector_root),'--bank',row['bank_root'],'--bank-sha256',row['bank_sha256'],'--seed',str(row['seed']),'--out',str(out)]
    with out.with_suffix('.log').open('w') as log:
        proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            try:rc=proc.wait(timeout=240)
            except subprocess.TimeoutExpired:raise TimeoutError('Collector matched reset exceeded wall watchdog')
            if rc:raise RuntimeError(f'Collector matched reset failed ({rc}); preserve {out.with_suffix(".log")}')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=35)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15)
    proof=json.loads((out/'matched-reset-proof.json').read_text())
    assert proof['status']=='PASS' and proof['seed']==row['seed'] and proof['source_receipt_sha256']==row['source_receipt_sha256']
    assert proof['protocol']==('collector-matched-reset-v1' if row['suite'] in SEEN_SUITES else 'collector-sealed-postsample-v1') and proof['finalization']['passed']
    assert proof['suite']==row['suite'] and proof['bank_sha256']==row['bank_sha256'] and proof['reference_run_sha256']==row['reference_run_sha256']
    receipt_path=out/'attempt-0000/receipt.json';assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==proof['receipt_sha256'];prepared=json.loads(receipt_path.read_text())
    return dict(proof,path=str(out),prepared_snapshot=prepared['prepared_snapshot'],training_first_frame_offset_ns=prepared['training_first_frame_offset_ns'],proof_sha256=hashlib.sha256((out/'matched-reset-proof.json').read_bytes()).hexdigest(),post_cleanup_native_bootstrap_frames=1,post_cleanup_reteleport=False)

def reset_scene(transport,repo,collector_root,row,out):
    """One entrance for qualification and policy evaluation's matched reset."""
    import numpy as np
    from collect import mcp
    repo=Path(repo)
    pre,pre_retired=install_fenced_boundary(transport,lambda:mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().restore('+repr(dict(state=row['state'],drive_targets=row['recorded_drive_targets'],bodies=row['bodies']))+',arm=True,restore_drives=True)'}))
    proof=prepare_scene(repo,collector_root,row,out)
    # Cleanup changes only the epoch and idle cap. No teleport is permitted
    # after normal collector preparation; retain its rendered scene history.
    mcp('run_python',{'code':(repo/'linux-env/scripts/collect/configure_clock.py').read_text()+'\nresult=configure(30)'})
    reset,rejoin_retired=install_fenced_boundary(transport,lambda:mcp('run_python',{'code':'import collection_runtime as c\nresult=c.install("collect")'}))
    before=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().snapshot()'});prepared=proof['prepared_snapshot']
    assert abs(before['sim_ns']-prepared['sim_ns'])<=1000,'Physics advanced during collector cleanup/rejoin'
    assert np.max(abs(np.asarray(before['state'])-prepared['state']))<=1e-9
    assert np.max(abs(np.asarray(before['drive_targets'])-prepared['drive_targets']))<=1e-9
    assert max(abs(x-y) for name in before['bodies'] for x,y in zip(before['bodies'][name]['position'],prepared['bodies'][name]['position']))<=1e-8
    proof['parent_prebootstrap_snapshot']=before
    mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read();after=transport.session.current
    assert after['epoch']==reset['epoch'] and abs((after['sim_ns']-before['sim_ns'])-1e9/30)<=1000
    proof.update(first_observation_sim_ns=after['sim_ns'],first_observation_epoch=after['epoch'],measured_first_frame_offset_ns=after['sim_ns']-prepared['sim_ns'])
    assert transport.sensors.failure is None,'Parent sensor ambiguity after concurrent collector preparation'
    proof['parent_sensors']=dict(failure=None,duplicate_messages=dict(transport.sensors.duplicate_messages),rerender_messages=dict(transport.sensors.rerender_messages),stale_camera_messages=dict(transport.sensors.stale_camera_messages))
    proof['retired_parent_buffers']=dict(before_worker=pre_retired,after_worker=rejoin_retired)
    return reset,proof

def require_qualification(root,host,collector_root,bank_sha256,source_dir,*,suite='seen-training-home',reference_sha256=None,reference_sha256_by_seed=None):
    root=Path(root);ledger=json.loads((root/'window-ledger.json').read_text());summary=json.loads((root/'qualification/summary.json').read_text())
    assert ledger['status']=='matched-reset-qualified' and ledger['cleanup_exit_code']==0
    assert ledger['host_profile']==host and summary['bank_sha256']==bank_sha256 and Path(summary['reset_collector_root']).resolve()==Path(collector_root).resolve()
    seeds=list(range(530000,530005)) if suite=='seen-training-home' else list(SEEN15_SEEDS) if suite=='seen-training-home-15' else list(range(900000,900030))
    assert suite in (*SEEN_SUITES,'heldout-bank-v2')
    assert summary['status']=='PASS' and summary['suite']==suite and summary['seeds']==seeds
    assert summary['episodes']==len(seeds) and summary['commands']==10*len(seeds)
    if suite in ('seen-training-home-15','rule-seen-training-home-15'):
        assert reference_sha256 is None and reference_sha256_by_seed is not None,'Seen fifteen requires per-seed reference binding'
        assert set(reference_sha256_by_seed)==set(map(str,seeds))
        assert summary['reference_run_sha256'] is None and summary['reference_run_sha256_by_seed']==reference_sha256_by_seed
    else:
        assert reference_sha256_by_seed is None
        if reference_sha256 is not None:assert summary['reference_run_sha256']==reference_sha256
    from native_environment import qualification_sources
    assert qualification_sources(source_dir)==ledger['source_sha256'],'Runtime source differs from live qualification'
    if suite=='heldout-bank-v2':
        from native_bank import SEEN_BANK_SHA
        control=summary['bootstrap_control']
        assert control['bank_sha256']==SEEN_BANK_SHA,'Bootstrap control must use the fixed training-home bank'
        actual=require_qualification(control['path'],host,collector_root,control['bank_sha256'],source_dir,suite='seen-training-home',reference_sha256=summary['reference_run_sha256'])
        assert actual==control,'Bootstrap control changed since held-out qualification'
    else:assert summary.get('bootstrap_control') is None
    return dict(path=str(root),bank_sha256=bank_sha256,ledger_sha256=hashlib.sha256((root/'window-ledger.json').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((root/'qualification/summary.json').read_bytes()).hexdigest())
