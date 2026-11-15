#!/usr/bin/env python3
"""Report a frozen MAIN window with every planned scene in the denominator."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
from report_phase3 import validity
from report_results import wilson
from phase3 import manifest,BANK_SHA256,COLORS

FLAGS=('success','grasp_success','correct_cup','target_wrong_cup','wrong_color_lift','wrong_block_placement','any_placement','target_any_placement')

def percentile(values,q):
    return sorted(values)[max(0,math.ceil(q*len(values))-1)] if values else None

def mean(values):
    return statistics.mean(values) if values else None

def select_examples(rows):
    selected=[]
    for color in COLORS:
        rs=[r for r in rows if r['color']==color and r.get('result_file') and r.get('valid')]
        successes=[r for r in rs if r['success']];failures=[r for r in rs if not r['success']]
        if successes:selected.append((min(successes,key=lambda r:r['time_to_place_s'] if r.get('time_to_place_s') is not None else math.inf),'success'))
        if failures:selected.append((min(failures,key=lambda r:r['closing_distance_m'] if r.get('closing_distance_m') is not None else math.inf),'failure: closest first closing'))
        if not successes and len(failures)>1:
            remaining=[r for r in failures if r['seed']!=selected[-1][0]['seed']]
            selected.append((max(remaining,key=lambda r:r['closing_distance_m'] if r.get('closing_distance_m') is not None else -1),'failure: farthest first closing'))
    invalid=next((r for r in rows if not r.get('valid') and r.get('result_file')),None)
    if invalid is not None:selected.append((invalid,'invalid: '+invalid['validity_reasons']))
    return selected

def collect(root,expected_step=10000):
    expected={r['seed']:r for r in manifest()['per_checkpoint']}; found={}; configs=[];attempted={};result_paths=[]
    attempts=[p for p in sorted(root.glob('attempt-*')) if p.is_dir()]
    if not attempts:raise ValueError('Expected window root containing attempt-* directories')
    for attempt in attempts:
        config=json.loads((attempt/'run_config.json').read_text())
        if config.get('scene_bank_sha256')!=BANK_SHA256 or config.get('inference_mode')!='eager':raise ValueError('Changed bank/backend')
        if Path(config['pretrained']).name!='pretrained_model' or int(Path(config['pretrained']).parent.name)!=expected_step:
            raise ValueError('Unexpected checkpoint step')
        if configs and any(config[k]!=configs[0][k] for k in ('pretrained','scene_bank_sha256','source_sha256','inference_mode','fork_commit')):
            raise ValueError('Mixed model or source contracts')
        if (root/'preflight.json').exists():
            proof=json.loads((root/'preflight.json').read_text())
            if config['source_sha256']!=proof['source_sha256']:raise ValueError('Source differs from qualification preflight')
        configs.append(config)
        entries=json.loads((attempt/'episode-manifest.json').read_text())
        if not isinstance(entries,list):raise ValueError('Expected episode manifest array')
        if len({e['seed'] for e in entries})!=len(entries):raise ValueError('Duplicate seed in attempt manifest')
        for e in entries:
            if e['seed'] not in expected or any(e.get(k)!=expected[e['seed']][k] for k in ('suite','color','phrase','task','rtc')):
                raise ValueError('Wrong planned MAIN scene contract')
        for ep in sorted(p for p in attempt.glob('ep*') if p.is_dir()):
            try:e=entries[int(ep.name[2:])]
            except (ValueError,IndexError):raise ValueError('Unmapped episode directory: '+str(ep))
            seed=e['seed']
            if seed in attempted:raise ValueError(f'Unexpected or duplicate seed attempt: {seed}')
            attempted[seed]=ep
            if (ep/'result.json').exists():result_paths.append((ep/'result.json',e))
    for path,e in result_paths:
        raw=json.loads(path.read_text()); seed=raw['seed']
        if seed!=e['seed']:raise ValueError('Result seed does not match its episode manifest entry')
        if any(raw.get(k)!=e[k] for k in ('suite','color','phrase','task')) or raw.get('rtc_enabled') is not True:
            raise ValueError(f'Wrong MAIN contract for seed {seed}')
        reasons=validity(raw)
        health=path.parent.parent/'window-health.json'
        if not health.exists() or not json.loads(health.read_text()).get('passed'):reasons.append('missing-or-failed-window-health')
        c=raw.get('client',{}); costs=c.get('send_action_costs_ms',[]);trace=c.get('clock_dispatch_trace',[])
        jitter=[abs(v['jitter_ms']) for v in trace];loads=[(raw.get('host_load_start') or [None])[0],(raw.get('host_load_end') or [None])[0]]
        major_start=raw.get('isaac_major_faults_start');major_end=raw.get('isaac_major_faults_end')
        row={k:raw.get(k) for k in ('seed','color','task','outcome','closing_distance_m','closest_target_distance_m','closing_motion_ge2deg','time_to_grasp_s','time_to_place_s','sim_time_s','wall_time_s','total_episode_wall_s','actions_sent','chunks','effective_action_fps','empty_ticks_after_startup','empty_ticks_pct','first_chunk_s','request_chunk_mean_ms','request_chunk_p95_ms','request_queue_depth_mean','chunk_size_threshold','actions_per_chunk','sim_cap_hz','failure_note')}
        row.update(valid=not reasons,validity_reasons=';'.join(reasons),reported_outcome=raw['outcome'] if not reasons else 'invalid',result_file=str(path.relative_to(root)),video_file=str((path.parent/'viewport.mp4').relative_to(root)),rtc_enabled=True,
            effective_wall_action_fps=c.get('effective_wall_action_fps'),missed_sim_slots=c.get('missed_action_ticks_after_startup'),
            send_median_ms=percentile(costs,.5),send_p95_ms=percentile(costs,.95),jitter_p95_ms=percentile(jitter,.95),jitter_max_ms=max(jitter) if jitter else None,
            load1_start=loads[0],load1_end=loads[1],isaac_major_faults_delta=major_end-major_start if major_end is not None and major_start is not None else None,
            starved=bool(raw['empty_ticks_after_startup']) if raw.get('empty_ticks_after_startup') is not None else None,lifted_objects=json.dumps(raw.get('lifted_objects',[])),placements=json.dumps(raw.get('placements',[])))
        for flag in FLAGS:
            row['observed_'+flag]=raw.get(flag)
            row[flag]=bool(raw.get(flag)) and not reasons
        found[seed]=row
    for seed,e in expected.items():
        if seed not in found:
            found[seed]=dict(seed=seed,color=e['color'],task=e['task'],valid=False,reported_outcome='invalid' if seed in attempted else 'not-run',validity_reasons='missing-result' if seed in attempted else 'not-run',starved=None,**{flag:False for flag in FLAGS})
    return [found[s] for s in sorted(found)],configs[0] if configs else None

def summarize(rows):
    n=len(rows);valid=[r for r in rows if r['valid']]
    result=dict(n=n,valid=len(valid),invalid=sum(r['reported_outcome']=='invalid' for r in rows),not_run=sum(r['reported_outcome']=='not-run' for r in rows),starved=sum(r.get('starved') is True for r in rows))
    for f in FLAGS:
        k=sum(r[f] for r in rows);result[f]=dict(k=k,n=n,rate=k/n,wilson95=wilson(k,n))
    placed=sum(r['target_any_placement'] for r in rows);correct=sum(r['correct_cup'] for r in rows)
    result['correct_cup_given_target_placement']=dict(k=correct,n=placed,rate=correct/placed if placed else None)
    for f in ('closing_distance_m','closest_target_distance_m','empty_ticks_pct','effective_action_fps','effective_wall_action_fps','wall_time_s','total_episode_wall_s','send_p95_ms','send_median_ms','jitter_p95_ms','jitter_max_ms','request_chunk_mean_ms','request_chunk_p95_ms','first_chunk_s','load1_start','load1_end'):
        vals=[r[f] for r in rows if r.get(f) is not None]
        vv=[r[f] for r in valid if r.get(f) is not None]
        result[f]=dict(n=len(vals),mean=mean(vals),median=percentile(vals,.5),min=min(vals) if vals else None,max=max(vals) if vals else None,valid_only_n=len(vv),valid_only_mean=mean(vv))
    vk=sum(r['success'] for r in valid)
    result['valid_only_success']=dict(k=vk,n=len(valid),wilson95=wilson(vk,len(valid)) if valid else None)
    return result

def report(root,out,expected_step=10000):
    rows,config=collect(root,expected_step);out.mkdir(parents=True,exist_ok=True)
    summary=dict(overall=summarize(rows),per_color={c:summarize([r for r in rows if r['color']==c]) for c in COLORS},contract=config,
        denominator='All 30 planned seeds. Invalid/non-run rows are distinct outcomes and cannot contribute confirmed successes. Starvation is overlapping, not an exclusion. Timing/distance averages are diagnostic over all observed rows, including invalid rows; valid-only averages are separate.',
        limitations='Marginal Wilson intervals; ten scenes per colour. Layout pairing only; policy flow RNG not paired. One checkpoint/arm cannot establish an RTC or training-step effect.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    (out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows,summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=int,default=10000);a=p.parse_args()
    rows,summary=report(a.root,a.out,a.step);print(json.dumps(summary['overall'],indent=2))
if __name__=='__main__':main()
