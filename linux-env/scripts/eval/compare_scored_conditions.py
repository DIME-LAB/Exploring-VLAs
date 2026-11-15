"""Compare paired MAIN layouts across checkpoint or RTC conditions."""
from pathlib import Path
import argparse,csv,json,statistics
from checkpoint_identity import checkpoint_label

METRICS=('closing_distance_m','min_tcp_target_m','min_tcp_other_m','tcp_dwell_pct','closing_time_sim_s','target_nearer_at_closing','ever_closed_ge2deg')
CONTRACT=('scene_bank_sha256','source_sha256','inference_mode','fork_commit','fps','actions_per_chunk','chunk_size_threshold','aggregate_fn','joint_states_topic','cameras_json','sim_time','sim_rate_hz','rtc_execution_horizon','rtc_max_guidance_weight','rtc_schedule')
def compare(left,right):
    summaries=[json.loads((p/'summary.json').read_text()) for p in (left,right)]
    for k in CONTRACT:
        assert summaries[0]['contract'][k]==summaries[1]['contract'][k],f'Unmatched deployment: {k}'
    changed=[summaries[0]['contract'][k]!=summaries[1]['contract'][k] for k in ('pretrained','rtc_enabled')]
    assert sum(changed)==1,'Compare exactly one changed factor: checkpoint or RTC'
    rows=[{r['seed']:r for r in json.loads((p/'episodes.json').read_text())} for p in (left,right)]
    physical=[{r['seed']:r for r in json.loads((p/'behaviour/episodes.json').read_text())} for p in (left,right)]
    assert all(set(r)==set(range(900000,900030)) for r in rows+physical)
    pairs=[]
    for seed in sorted(rows[0]):
        x,y=rows[0][seed],rows[1][seed];bx,by=physical[0][seed],physical[1][seed]
        assert x['color']==y['color'] and x['task']==y['task']
        p=dict(seed=seed,color=x['color'],valid_before=x['valid'],valid_after=y['valid'],valid_both=x['valid'] and y['valid'],outcome_before=x['reported_outcome'],outcome_after=y['reported_outcome'],success_before=x['success'],success_after=y['success'],success_difference=int(y['success'])-int(x['success']))
        for key in METRICS:
            a,b=bx.get(key),by.get(key);p[key+'_before']=a;p[key+'_after']=b;p[key+'_difference']=float(b)-float(a) if a is not None and b is not None else None
        for key in ('empty_ticks_pct','effective_action_fps'):
            a,b=x.get(key),y.get(key);p[key+'_before']=a;p[key+'_after']=b;p[key+'_difference']=b-a if a is not None and b is not None else None
        pairs.append(p)
    valid=[r for r in pairs if r['valid_both']]
    result=dict(planned_n=len(pairs),valid_both_n=len(valid),operational_success_difference=statistics.mean(r['success_difference'] for r in pairs),discordants_favour_after=sum(r['success_difference']==1 for r in valid),discordants_favour_before=sum(r['success_difference']==-1 for r in valid))
    for key in (*METRICS,'empty_ticks_pct','effective_action_fps'):
        values=[r[key+'_difference'] for r in valid if r[key+'_difference'] is not None]
        result[key+'_paired_difference']=dict(n=len(values),mean=statistics.mean(values) if values else None)
    result['conditions']=[]
    for root,s,b in zip((left,right),summaries,physical):
        closed=[r for r in b.values() if r.get('ever_closed_ge2deg')]
        result['conditions'].append(dict(root=str(root),checkpoint=s['contract']['pretrained'],rtc=s['contract']['rtc_enabled'],summary=s['overall'],closed_n=len(closed),target_nearest_at_closure_n=sum(r.get('target_nearer_at_closing') is True for r in closed)))
    return pairs,result

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();pairs,stats=compare(a.before,a.after);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (a.out/'paired-seeds.csv').open('w') as f:w=csv.DictWriter(f,list(pairs[0]));w.writeheader();w.writerows(pairs)
    labels=[f"{checkpoint_label(c['checkpoint'])} RTC {'on' if c['rtc'] else 'off'}" for c in stats['conditions']]
    table=['| Condition | Success (95% Wilson) | Target grasp | Wrong-block lift | Requested colour nearest / closures | Closing mean (mm; N) | Dwell (%) | Invalid / starved | Empty ticks (%) |','|---|---|---|---|---|---|---|---|---|']
    for label,c in zip(labels,stats['conditions']):
        s=c['summary'];i=s['success'];lo,hi=i['wilson95'];close=s['closing_distance_m']
        mean='—' if close['mean'] is None else f"{1000*close['mean']:.2f}"
        table.append(f"| {label} | {i['k']}/{i['n']} ({100*lo:.1f}–{100*hi:.1f}%) | {s['grasp_success']['k']}/30 | {s['wrong_color_lift']['k']}/30 | {c['target_nearest_at_closure_n']}/{c['closed_n']} | {mean}; {close['n']} | {statistics.mean(r['tcp_dwell_pct_'+('before' if label==labels[0] else 'after')] for r in pairs):.2f} | {s['invalid']} / {s['starved']} | {s['empty_ticks_pct']['mean']:.2f} |")
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for i,c in enumerate(stats['conditions']):
        s=c['summary']['success'];v=100*s['rate'];lo,hi=s['wilson95'];axes[0].errorbar(i,v,yerr=[[v-100*lo],[100*hi-v]],fmt='o',capsize=5);axes[0].text(i,100*hi+2,f"{s['k']}/30",ha='center')
    axes[0].set(ylim=(0,100),ylabel='Confirmed success (%)',title='Marginal 95% Wilson intervals')
    for ax,key,scale,title in zip(axes[1:],('closing_distance_m','tcp_dwell_pct'),(1000,1),('First closure distance (mm)','Low-motion dwell (%)')):
        for r in pairs:
            x,y=r[key+'_before'],r[key+'_after']
            if x is None or y is None:continue
            ax.plot([0,1],[x*scale,y*scale],color='#777' if r['valid_both'] else '#bd4444',alpha=.4,ls='-' if r['valid_both'] else '--')
        ax.set(ylabel=title)
    for ax in axes:ax.set_xticks([0,1],labels);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(a.out/'comparison.png',dpi=170);plt.close(fig)
    def delta(key,scale=1):
        x=stats[key+'_paired_difference'];return 'undefined (N=0)' if x['mean'] is None else f"{scale*x['mean']:+.2f} (N={x['n']})"
    text=f'''{labels[1]} has {stats['conditions'][1]['summary']['success']['k']}/30 confirmed successes versus {stats['conditions'][0]['summary']['success']['k']}/30 for {labels[0]}, on the same thirty held-out layouts. Both episodes pass timing validity on {stats['valid_both_n']} shared seeds. All planned scenes remain in outcome denominators; invalid observations cannot be confirmed successes.

{chr(10).join(table)}

![Paired results](comparison.png)

After minus before on valid shared pairs: first-closing target distance {delta('closing_distance_m',1000)} mm; minimum TCP-to-target distance {delta('min_tcp_target_m',1000)} mm; dwell {delta('tcp_dwell_pct')} percentage points; empty ticks {delta('empty_ticks_pct')} percentage points. Missing closures remain missing, so distance comparisons require both closures and give their own denominator. All-observed descriptive means in the table include invalid attempts; the paired means require both attempts valid.

Confirmed-success operational difference: {100*stats['operational_success_difference']:+.1f} percentage points. Valid-pair discordants favouring after/before: {stats['discordants_favour_after']}/{stats['discordants_favour_before']}. Intervals are marginal, paired deltas descriptive, and MAIN flow-noise was not paired. Thirty layouts and zero/few successes do not establish equivalence or a precise effect. These runs were sequential and host load was not randomized. See per-episode boundary load and timing in each condition report. Source, bank, inference mode, camera/state contract, action/re-observation rates and native queue aggregation are matched; RTC or checkpoint weights are the intended factors.

Closing distance is measured fixed-jaw TCP to target centre at first ≥2° closure, not contact or placement. Low-motion dwell uses half-second windows within a 5 mm TCP box diagonal and 0.5° gripper range. Short/long curved trajectories can share a net direction; these physical measures complement the separate CPU language/noise probes.

[paired-seeds.csv](paired-seeds.csv) contains every seed, validity, outcome, closure status, closing/closest distances, dwell, empty ticks and effective Hz. [summary.json](summary.json) records per-metric paired denominators and both full condition summaries. Individual videos, raw telemetry and failure characterization remain in the condition deliveries.
'''
    (a.out/'README.md').write_text(text);print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
