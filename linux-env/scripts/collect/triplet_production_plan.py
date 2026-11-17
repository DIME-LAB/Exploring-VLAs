"""Deterministic triplet waves, replacement chains and operator stop guards.

Outcomes supplied to this reducer must have their source artifacts verified by
the runner. Checkpoint counters/selected flags are never planning authority.
"""
import copy,re
from contract import (COUNTERFACTUAL_TRAIN_BASE_SEEDS,COUNTERFACTUAL_HELDOUT_BASE_SEEDS,
    COUNTERFACTUAL_TRAIN_RESERVE_SEEDS,COUNTERFACTUAL_HELDOUT_RESERVE_SEEDS,EVAL_SEEDS)

BASE=dict(train=COUNTERFACTUAL_TRAIN_BASE_SEEDS,heldout=COUNTERFACTUAL_HELDOUT_BASE_SEEDS)
RESERVE=dict(train=COUNTERFACTUAL_TRAIN_RESERVE_SEEDS,heldout=COUNTERFACTUAL_HELDOUT_RESERVE_SEEDS)
POLICY=dict(version=2,instances=[3,4,5,6],targets=dict(train=120,heldout=20),
    reserve=dict(train=list(RESERVE['train']),heldout=list(RESERVE['heldout'])),
    replacement_order='complete wave, then sort original requested slots within train/heldout; ascending unused reserve seeds',
    exact_refusal_fraction_max=.10,reserve_fraction_max=.50,reserve_stop_at_half=True,
    media=dict(exact_first_physical_hash=True,exact_first_state=True,min_each_camera_psnr_db=40.,different_first50_labels=True))


def validate_ranges():
    groups=[set(BASE[s]) for s in BASE]+[set(RESERVE[s]) for s in RESERVE]
    if sum(map(len,groups))!=len(set().union(*groups)):raise ValueError('Triplet seed ranges overlap')
    occupied=set(range(330000,350100))|set(range(400000,570000))|set(range(600000,700000))|set(EVAL_SEEDS)
    if set().union(*groups)&occupied:raise ValueError('Triplet seeds overlap existing collection/evaluation ranges')
    return True


def job(split,requested,chain,wave):
    if requested not in BASE[split] or chain[0]!=requested:raise ValueError('Unknown original requested slot')
    if chain!=[requested] and any(seed not in RESERVE[split] for seed in chain[1:]):raise ValueError('Undeclared replacement seed')
    ordinal=BASE[split].index(requested)+(0 if split=='train' else len(BASE['train']))
    return dict(split=split,requested_layout_seed=requested,layout_seed=chain[-1],
        replaces_layout_seed=chain[-2] if len(chain)>1 else None,replacement_chain=list(chain),wave=wave,
        instance=POLICY['instances'][ordinal%len(POLICY['instances'])])


def guards(outcomes,allocations):
    exact=sum(r['reason']=='exact_identity' for r in outcomes)
    if outcomes and exact/len(outcomes)>POLICY['exact_refusal_fraction_max']:
        raise ValueError('Production stop: exact-identity refusals exceed 10%: '+str(dict(exact=exact,attempted=len(outcomes))))
    for split in BASE:
        used=sum(r['requested_layout_seed'] in BASE[split] for r in allocations)
        if used>=len(RESERVE[split])*POLICY['reserve_fraction_max']:
            raise ValueError('Production stop: half '+split+' reserve would be consumed')
    # Derive per-instance streaks from verified, ordered outcomes, including
    # earlier waves. Starting a replacement worker never resets this history.
    streak={}
    for row in outcomes:
        if 'requested_layout_seed' not in row:continue
        split=row['split'];ordinal=BASE[split].index(row['requested_layout_seed'])+(0 if split=='train' else len(BASE['train']))
        slot=POLICY['instances'][ordinal%len(POLICY['instances'])]
        previous,count=streak.get(slot,(None,0));reason=row['reason']
        if reason=='accepted':streak[slot]=(None,0);continue
        count=count+1 if previous==reason else 1;streak[slot]=(reason,count)
        if count>=3:raise ValueError('Production stop: three consecutive same-reason refusals on instance '+str(slot)+': '+reason)


def initial_wave():
    validate_ranges()
    return [job(split,seed,[seed],0) for split in BASE for seed in BASE[split]]


def reduce_waves(waves):
    """Rebuild all memberships from complete source-verified waves, not a checkpoint."""
    expected=initial_wave();selected={};failed=[];allocations=[];all_outcomes=[];consumed={s:0 for s in BASE}
    for number,wave in enumerate(waves):
        if not expected:raise ValueError('Ledger continued after all requested groups were selected')
        if wave['jobs']!=expected:raise ValueError('Wave jobs differ from deterministic seed/slot plan')
        outcomes=wave['outcomes'];by_seed={}
        for result in outcomes:
            required={'split','layout_seed','requested_layout_seed','status','reason','proof_path','proof_sha256'}
            if not isinstance(result,dict) or set(result)!=required:raise ValueError('Incomplete or conflicting group disposition fields')
            key=(result['split'],result['layout_seed'])
            if key in by_seed:raise ValueError('Duplicate group outcome')
            if (result['status'] not in ('selected','refused','terminal') or not isinstance(result['proof_sha256'],str) or
                    not re.fullmatch('[0-9a-f]{64}',result['proof_sha256'])):raise ValueError('Unverified group disposition')
            if (not isinstance(result['proof_path'],str) or not result['proof_path'] or
                    not isinstance(result['reason'],str) or not result['reason']):raise ValueError('Missing group proof/reason')
            if (result['status']=='selected')!=(result['reason']=='accepted'):raise ValueError('Disposition status/reason conflict')
            by_seed[key]=result
        if set(by_seed)!={(j['split'],j['layout_seed']) for j in expected}:raise ValueError('Incomplete or orphan wave outcomes')
        refused=[]
        for planned in expected:
            result=by_seed[(planned['split'],planned['layout_seed'])]
            if result['requested_layout_seed']!=planned['requested_layout_seed']:raise ValueError('Outcome changed its requested slot')
            if result['status']=='terminal':raise ValueError('Production stop: terminal pipeline fault: '+result['reason'])
            all_outcomes.append(result)
            if result['status']=='selected':selected[(planned['split'],planned['requested_layout_seed'])]=dict(planned,proof_path=result['proof_path'],proof_sha256=result['proof_sha256'])
            else:
                failed.append(dict(planned,proof_path=result['proof_path'],proof_sha256=result['proof_sha256'],disposition=result['reason']))
                refused.append((planned,result))
        guards(all_outcomes,allocations)
        expected=[]
        for planned,result in sorted(refused,key=lambda pair:(list(BASE).index(pair[0]['split']),pair[0]['requested_layout_seed'])):
            split=planned['split'];offset=consumed[split]
            if offset>=len(RESERVE[split]):raise ValueError('Replacement reserve exhausted: '+split)
            seed=RESERVE[split][offset];consumed[split]+=1
            allocation=dict(wave=number+1,requested_layout_seed=planned['requested_layout_seed'],
                replaces_layout_seed=planned['layout_seed'],layout_seed=seed,failed_proof_sha256=result['proof_sha256'])
            allocations.append(allocation)
            expected.append(job(split,planned['requested_layout_seed'],planned['replacement_chain']+[seed],number+1))
        guards(all_outcomes,allocations)
    return dict(policy=copy.deepcopy(POLICY),next_wave=expected,selected=list(selected.values()),failed_groups=failed,
        replacement_allocations=allocations,attempted_groups=len(all_outcomes),complete=not expected and len(selected)==sum(map(len,BASE.values())))
