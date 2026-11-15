#!/usr/bin/env python3
"""Report Phase3 exact-string and language suites without pooling their scores."""
import argparse,csv,json,math,statistics
from pathlib import Path
from report_results import wilson
from phase3 import manifest,COLORS,BANK_SHA256


def validity(row):
    c=row.get('client',{});costs=sorted(c.get('send_action_costs_ms',[]))
    reasons=[]
    if row.get('outcome') not in ('success','grasped-not-placed','no-grasp'):reasons.append('episode-error')
    if row.get('clock_basis')!='simulation' or c.get('clock_basis')!='simulation':reasons.append('wrong-clock')
    if row.get('sim_cap_hz')!=30:reasons.append('cap-readback')
    if c.get('missed_action_ticks_after_startup')!=0:reasons.append('missed-sim-slots')
    if c.get('nonmonotonic_action_steps')!=0:reasons.append('nonmonotonic-action')
    if not costs or costs[math.ceil(.95*len(costs))-1]>=5:reasons.append('send-cost')
    if c.get('observe_every_actions')!=10:reasons.append('wrong-reobserve-cadence')
    if not c.get('clock_dispatch_trace'):reasons.append('missing-clock-trace')
    cadence=c.get('observation_cadence_events',[])
    if not cadence:reasons.append('missing-cadence-trace')
    for event in cadence[1:]:
        if event.get('actions_since_request',0)<10 and event.get('empty_slots_since_request',0)<10:
            reasons.append('early-reobserve');break
    chunks=c.get('chunk_events',[])
    if row.get('rtc_enabled') and len(chunks)>=2 and not any(e.get('eval_rtc',{}).get('prefix_steps',0)>0 for e in chunks if e.get('eval_rtc')):
        reasons.append('missing-guided-prefix')
    return reasons


def summarize(rows):
    n=len(rows)
    result=dict(n=n)
    for name in ('success','grasp_success','correct_cup','any_placement','target_any_placement','target_wrong_cup','wrong_block_placement'):
        k=sum(bool(row.get(name)) for row in rows)
        result[name]=dict(k=k,n=n,rate=k/n if n else None,wilson95=wilson(k,n) if n else None)
    placed=sum(row.get('target_any_placement',False) for row in rows)
    correct=sum(row.get('correct_cup',False) for row in rows)
    result['correct_cup_given_target_placement']=dict(k=correct,n=placed,rate=correct/placed if placed else None)
    for field in ('closing_distance_m','closest_target_distance_m','empty_ticks_pct','effective_action_fps','wall_time_s','sim_dispatch_jitter_p95_ms'):
        values=[row[field] for row in rows if row.get(field) is not None]
        result[field]=dict(n=len(values),mean=statistics.mean(values) if values else None)
    return result


def report(root):
    grouped={};invalid=[];rows=[];seen=set();contract=None
    for file in sorted(root.rglob('result.json')):
        row=json.loads(file.read_text())
        if row.get('clock_basis')!='simulation':continue
        config=json.loads((file.parent.parent/'run_config.json').read_text())
        current=dict(scene_bank_sha256=config.get('scene_bank_sha256'),source_sha256=config.get('source_sha256'),inference_mode=config.get('inference_mode'),fork_commit=config.get('fork_commit'))
        if current['scene_bank_sha256']!=BANK_SHA256 or current['inference_mode']!='eager':
            raise ValueError('Run differs from frozen bank/backend protocol: '+str(file))
        if contract is None:contract=current
        elif current!=contract:raise ValueError('Compared rows use different source/bank/backend contracts')
        health=file.parent.parent/'window-health.json'
        if not health.exists() or not json.loads(health.read_text()).get('passed'):
            invalid.append(dict(source=str(file),reasons=['missing-or-failed-window-health']));continue
        model=config['pretrained'];key=(model,row['suite'],row['seed'],row['color'],row['phrase'],row['rtc_enabled'])
        if key in seen:raise ValueError('Duplicate episode key; explicitly archive superseded attempts: '+repr(key))
        seen.add(key);row['model']=model;row['source']=str(file)
        reasons=validity(row)
        if reasons:invalid.append(dict(source=str(file),reasons=reasons));continue
        rows.append(row)
        g=(model,row['suite'],row['color'],row['phrase'],row['rtc_enabled'])
        grouped.setdefault(g,[]).append(row)
    groups=[dict(model=g[0],suite=g[1],color=g[2],phrase=g[3],rtc=g[4],**summarize(rs)) for g,rs in grouped.items()]
    candidates=[]
    expected={(r['seed'],r['color'],r['task']) for r in manifest()['per_checkpoint']}
    for model in sorted({row['model'] for row in rows}):
        main=[r for r in rows if r['model']==model and r['suite']=='main' and r['rtc_enabled']]
        if {(r['seed'],r['color'],r['task']) for r in main}==expected and len(main)==30:
            # The checkpoint directory's numeric parent identifies the step.
            step=int(Path(model).parent.name)
            candidates.append((sum(bool(r['success']) for r in main),step,model))
    cross=[]
    mains={}
    for row in rows:
        if row["suite"]=="main" and row["rtc_enabled"]:
            mains.setdefault(row["model"],{})[(row["seed"],row["color"],row["task"])]=row
    ordered=sorted(mains,key=lambda model:int(Path(model).parent.name))
    for before,after in zip(ordered,ordered[1:]):
        for key in sorted(mains[before].keys() & mains[after].keys()):
            a,b=mains[before][key],mains[after][key]
            cross.append(dict(before=before,after=after,seed=key[0],color=key[1],success_difference=int(b["success"])-int(a["success"]),closing_distance_difference_m=b["closing_distance_m"]-a["closing_distance_m"] if a["closing_distance_m"] is not None and b["closing_distance_m"] is not None else None))
    paired=[]
    baseline={(r['model'],r['seed'],r['color']):r for r in rows if r['suite']=='main' and r['rtc_enabled']}
    for row in rows:
        if row['suite']=='main-rtc-off':
            other=baseline.get((row['model'],row['seed'],row['color']))
            if other:paired.append(dict(model=row['model'],seed=row['seed'],color=row['color'],success_off_minus_on=int(row['success'])-int(other['success']),closing_distance_off_minus_on_m=row['closing_distance_m']-other['closing_distance_m'] if row['closing_distance_m'] is not None and other['closing_distance_m'] is not None else None))
    return dict(groups=groups,invalid=invalid,paired_rtc=paired,paired_checkpoints=cross,best_checkpoint=max(candidates)[2] if candidates else None,
                limit='Wilson intervals are marginal, not paired confidence intervals. Paired differences are descriptive; report discordant counts and small N. Success requires requested-colour block in matching cup; correct-cup denominator is stated explicitly.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=report(a.root);a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
