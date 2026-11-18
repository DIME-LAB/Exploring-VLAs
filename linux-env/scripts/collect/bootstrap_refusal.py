"""Recompute one pre-writer bootstrap identity refusal from native evidence.

The collector remains terminal. Only the outer runner may reject this entire
layout after verifying its recording boundary, pipeline and healthy prefix.
"""
import ast,hashlib,json,math,numbers
from pathlib import Path
from contract import timing_audit
from episode_policy import disposition
from scene_identity import identity,require_identical,require_pose_agreement,pose_agreement,require_observation_drift,verify_pose_evidence,prior_agreements
from target_triplets import member
from triplet_artifacts import read,sha
import triplet_boundary as boundary


def physical_error(text):
    """Accept the producer's ValueError representation, never execute it."""
    for _ in range(3):
        try:node=ast.parse(text,mode='eval').body
        except (SyntaxError,ValueError,TypeError):return None
        if isinstance(node,ast.Constant) and isinstance(node.value,str):text=node.value;continue
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='ValueError' and
            len(node.args)==1 and not node.keywords and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)):
            prefix='Triplet physical agreement failed: '
            if node.args[0].value.startswith(prefix):return json.loads(node.args[0].value[len(prefix):])
        return None
    return None


def bind_failure(stored,recomputed):
    if not isinstance(stored,dict) or set(stored)!=set(recomputed) or stored.get('passed') is not False or recomputed['passed'] is not False:
        raise ValueError('Bootstrap failure lacks a locally failing physical verdict')
    fields=('policy','exact_bits','strict_differences')
    verify_pose_evidence({k:stored[k] for k in fields},{k:recomputed[k] for k in fields},'Bootstrap failure source bits/policy differ')
    # Errors/angles are diagnostics computed by independent interpreters.
    # Preserve their types and shapes; the local verdict above governs.
    def diagnostic(value,shape):
        if isinstance(shape,dict):
            if not isinstance(value,dict) or set(value)!=set(shape):raise ValueError('Bootstrap failure diagnostic fields differ')
            for k in shape:diagnostic(value[k],shape[k])
        elif isinstance(shape,list):
            if not isinstance(value,list) or len(value)!=len(shape):raise ValueError('Bootstrap failure diagnostic shape differs')
            for a,b in zip(value,shape):diagnostic(a,b)
        elif isinstance(shape,bool):
            if type(value) is not bool:raise ValueError('Bootstrap failure diagnostic boolean differs')
        elif not isinstance(value,numbers.Real) or isinstance(value,bool) or not math.isfinite(value):raise ValueError('Bootstrap failure diagnostic is not finite')
    diagnostic(stored['per_body'],recomputed['per_body'])
    return recomputed


def restored(record,anchor):
    if record.get('arm_restored') is not True or not record.get('command_baseline'):
        raise ValueError('Bootstrap refusal lacks a verified arm/drive restore')
    boundary.restore_admission(record['snapshot'],record,anchor)


def failure_comparison(observed,references,error):
    stored=physical_error(error)
    if stored is None:raise ValueError('Not a bootstrap physical-agreement exception')
    for reference in references:
        result=pose_agreement(observed,reference)
        if result['passed'] is False and result['exact_bits']==stored.get('exact_bits'):
            return bind_failure(stored,result)
    raise ValueError('Bootstrap exception does not match any failing native comparison')


def native_member(run,index,receipt,anchor,config,previous,previous_last):
    """Replay the existing native restore/observation/recording contracts."""
    phases=read(run/'triplet-boundaries'/f'attempt-{index:04d}.json')
    if phases['episode_id']!=receipt['episode_id'] or phases['layout_id']!=config['layout_id'] or phases['triplet_id']!=anchor['triplet_id']:
        raise ValueError('Bootstrap boundary belongs to another member/layout')
    phases=phases['phases']
    verify_pose_evidence(receipt['triplet_boundary_evidence'],phases,'Bootstrap boundary receipt differs')
    prime=phases['priming_restore']['restore'];final=phases['final_restore']['restore'];observed=phases['bootstrap']['snapshot']
    for one in (prime,final):restored(one,anchor['anchor'])
    for key,one in (('priming_restore',prime),('final_restore',final)):
        boundary.restore_prior(one,[p['phases'][key]['restore'] for p in previous])
    boundary.bootstrap(final['snapshot'],{k:v for k,v in final.items() if k!='snapshot'},observed)
    boundary.later(final['snapshot'],previous_last if previous_last is not None else anchor['anchor'])
    row=observed['tick_contract']['row']
    if (row['episode_id'],row['layout_id'],row['triplet_id'])!=(receipt['episode_id'],config['layout_id'],anchor['triplet_id']):
        raise ValueError('Bootstrap tick belongs to another member/layout')
    boundary.selections(observed,anchor['anchor']);drift=require_observation_drift(observed,anchor['anchor'])
    verify_pose_evidence(phases['bootstrap']['anchor_drift'],drift,'Bootstrap drift evidence differs')
    verify_pose_evidence(receipt['params'],member(config,index,anchor['anchor']),'Bootstrap task/target parameters differ')
    return dict(phases=phases,observed=observed,drift=drift)


def verify(run,config,receipts):
    run=Path(run);failed=receipts[-1];index=len(receipts)-1
    if index not in (1,2) or failed.get('exception_type')!='ValueError' or failed.get('capture_failures'):
        raise ValueError('Not an isolated later-member bootstrap identity failure')
    directory=run/f'attempt-{index:04d}'
    if {p.name for p in directory.iterdir()}!={'receipt.json','terminal-fault.json'}:
        raise ValueError('Bootstrap refusal has writer/frame or unknown member artifacts')
    if failed.get('end_cancel',{}).get('held') is not True or failed['end_cancel'].get('failure') or failed.get('retirement',{}).get('passed') is not True:
        raise ValueError('Bootstrap refusal retirement failed')
    anchor=read(run/'triplet-anchor.json')
    if (anchor['config']!=config or anchor['sampler_invocations']!=1 or identity(anchor['anchor'])!=anchor['anchor_sha256'] or
        identity(anchor['bootstrap'])!=anchor['bootstrap_sha256']):raise ValueError('Bootstrap refusal anchor changed')
    triplet_id=hashlib.sha256(json.dumps(dict(config=config,state=anchor['anchor_sha256']),sort_keys=True).encode()).hexdigest()
    if anchor['triplet_id']!=triplet_id:raise ValueError('Bootstrap refusal triplet identity differs')
    previous=[];previous_last=None;first=[]
    for i,receipt in enumerate(receipts):
        current=native_member(run,i,receipt,anchor,config,previous,previous_last)
        if i==index:break
        observed=current['observed'];require_pose_agreement(observed,anchor['bootstrap'])
        prior_agreements(observed,[p['observed'] for p in previous])
        if i==0:require_identical(observed,anchor['bootstrap'])
        rows=[json.loads(line) for line in read(run/f'attempt-{i:04d}/frames.jsonl',True).splitlines()]
        if (not rows or not timing_audit(rows,30)['passed'] or receipt.get('capture_failures') or
            receipt.get('finalization_errors') or receipt.get('terminal_publication_error') or
            receipt.get('camera_boundary',{}).get('passed') is not True or receipt.get('script_completed') is not True or
            receipt.get('score',{}).get('passed') is not True or receipt.get('score',{}).get('contract_passed') is not True or
            disposition(receipt,rows=rows).get('kind')!='accepted' or receipt.get('disposition',{}).get('kind')!='accepted'):
            raise ValueError('Bootstrap refusal prefix has a pipeline/quality fault')
        for row in rows:
            if (row['episode_id'],row['layout_id'],row['triplet_id'])!=(receipt['episode_id'],config['layout_id'],triplet_id):raise ValueError('Bootstrap refusal prefix frame identity differs')
        record=read(run/f'attempt-{i:04d}/first-recorded-native.json')
        boundary.verify_record(record,rows[0],observed,anchor['anchor'],receipt['episode_id'],triplet_id)
        prior_agreements(record['native'],first);first.append(record['native'])
        previous_last=boundary.from_row(rows[-1]);previous.append(current)
    if set(current['phases'])!={'priming_restore','final_restore','bootstrap'}:
        raise ValueError('Bootstrap refusal reached an unexpected or recorded phase')
    comparison=failure_comparison(current['observed'],[anchor['bootstrap']]+[p['observed'] for p in previous],failed.get('error'))
    marker=read(run/'terminal-fault.json');events=marker.get('events')
    if not events:raise ValueError('Bootstrap refusal lacks its durable native fault')
    if read(directory/'terminal-fault.json')!=marker:raise ValueError('Bootstrap refusal fault journals differ')
    for event in events:
        if event.get('source')!='attempt disposition':raise ValueError('Unrelated native event accompanied bootstrap refusal')
        bind_failure(physical_error(event.get('error')),comparison)
    return dict(version=1,kind='bootstrap_identity_refusal',training_forbidden=True,member=index,layout_id=config['layout_id'],triplet_id=triplet_id,
        anchor_sha256=sha(run/'triplet-anchor.json'),boundary_sha256=sha(run/'triplet-boundaries'/f'attempt-{index:04d}.json'),
        prefix_members=index,comparison=comparison)


def verify_evidence(stored,recomputed):
    if not isinstance(stored,dict) or set(stored)!=set(recomputed):raise ValueError('Bootstrap refusal evidence fields differ')
    bind_failure(stored['comparison'],recomputed['comparison'])
    verify_pose_evidence({k:v for k,v in stored.items() if k!='comparison'},
        {k:v for k,v in recomputed.items() if k!='comparison'},'Bootstrap refusal provenance differs')
