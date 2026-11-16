"""CPU-only target-rule analysis of the sealed 900 normal production episodes."""
import argparse,csv,hashlib,json,math,re,time
from collections import Counter
from pathlib import Path
from contract import COLORS,VISIBLE_TARGET_RULE,visible_target
from storage import CANONICAL_ROOT,atomic_json,confined


def read_bound(path,hashes,base):
    data=path.read_bytes();hashes[str(path.relative_to(base))]=hashlib.sha256(data).hexdigest()
    return json.loads(data)


def distribution(values):
    values=sorted(values)
    if not values:return dict(count=0)
    def percentile(p):
        i=(len(values)-1)*p;a=math.floor(i);b=math.ceil(i)
        return values[a]+(values[b]-values[a])*(i-a)
    return dict(count=len(values),min=min(values),p01=percentile(.01),p05=percentile(.05),p25=percentile(.25),
        median=percentile(.5),p75=percentile(.75),p95=percentile(.95),p99=percentile(.99),max=max(values),mean=sum(values)/len(values))


def summarize(rows):
    return dict(episodes=len(rows),nearest_xy_matches=sum(r['nearest_xy_matches'] for r in rows),
        visible_rule_matches=sum(r['visible_rule_matches'] for r in rows),
        unambiguous_matches=sum(r['unambiguous_matches'] for r in rows),ambiguous=sum(r['ambiguous'] for r in rows),
        margin_mm=distribution([1000*r['margin_m'] for r in rows]))


def analyze(root,expected=900):
    """Use the sealed index, not globbing: smoke and index folders are not batches."""
    started=time.monotonic();root=Path(root);hashes={}
    card=read_bound(root/'rtx6000-normal-900-index/data-card.json',hashes,root)
    if card.get('passed') is not True or card.get('accepted')!=expected:raise ValueError('Normal index is not complete')
    indexed=Counter((r['batch'],r['seed'],r['color']) for r in card['episodes'])
    if len(indexed)!=expected or any(v!=1 for v in indexed.values()):raise ValueError('Index episode identities are not bijective')
    rows=[];observed=Counter();seen_ids=set();seen_batches=set()
    for batch in card['batches']:
        name=batch['name']
        if not re.fullmatch(r'rtx6000-normal-i[3-6]-b\d{2}',name) or name in seen_batches:raise ValueError('Unexpected or repeated production batch')
        seen_batches.add(name);run=root/name/'normal'
        summary=read_bound(run/'summary.json',hashes,root)
        final=read_bound(run/'collector-finalization.json',hashes,root)
        if not summary.get('complete') or summary.get('end_marker')!='COLLECTION_FINISHED' or not final.get('passed') or not final.get('completed') or final.get('cleanup_errors'):
            raise ValueError('Batch is not affirmatively finalized: '+name)
        manifest=read_bound(run/'dataset/meta/collection_episodes.json',hashes,root)
        if len(manifest)!=batch['accepted'] or len(manifest)!=summary['episodes']:raise ValueError('Batch episode counts differ')
        for index,r in enumerate(manifest):
            params=r['params'];seed=params['seed'];color=params['color'];episode=r['episode_id']
            receipt=run/f"attempt-{params['trial']:04d}/receipt.json"
            raw=read_bound(receipt,hashes,root)
            if r!=dict(episode_index=index,**raw):raise ValueError('Raw receipt differs from exported manifest')
            if r['episode_index']!=index or not r.get('accepted') or params['recovery']!='none' or episode in seen_ids:raise ValueError('Invalid or repeated accepted episode')
            if params['task']!=f'Pick a {color} lego and place it in {color} cup':raise ValueError('Unexpected task')
            seen_ids.add(episode);observed[(name,seed,color)]+=1
            proof=visible_target(r['initial_physical'],color);recorded=params['object']
            if recorded not in {c['object'] for c in proof['candidates']}:raise ValueError('Recorded target is not a same-colour candidate')
            rows.append(dict(batch=name,episode_index=index,episode_id=episode,seed=seed,color=color,task=params['task'],
                receipt=str(receipt),dataset=str(run/'dataset'),initial_pose_source='receipt.initial_physical',recorded_target=recorded,
                nearest_xy_matches=recorded==proof['nearest_distance_object'],visible_rule_matches=recorded==proof['object'],
                unambiguous_matches=recorded==proof['object'] and not proof['ambiguous'],**proof))
    if observed!=indexed or len(rows)!=expected:raise ValueError('Analyzed episodes differ from the sealed index')
    return dict(version=1,rule=VISIBLE_TARGET_RULE,episodes=len(rows),passed=True,cpu_only=True,
        frame='fixed /World/base from initial_physical; XY centre distance, not 3-D distance',
        tie_policy='Within 5 mm use smaller base y, then object name; triplet collection refuses all such layouts',
        counts=summarize(rows),per_color={c:summarize([r for r in rows if r['color']==c]) for c in COLORS},
        per_batch={b:summarize([r for r in rows if r['batch']==b]) for b in sorted(seen_batches)},
        source_sha256=hashes,wall_seconds=time.monotonic()-started),rows


def publish(output,summary,rows):
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__)
    summary['analysis_source_sha256']={n:hashlib.sha256(source.with_name(n).read_bytes()).hexdigest()
        for n in ('analyze_target_rule.py','contract.py','pose_math.py')}
    atomic_json(output/'summary.json',summary);atomic_json(output/'episodes.json',rows)
    for name,field in [('rule-consistent','visible_rule_matches'),('unambiguous-rule-consistent','unambiguous_matches')]:
        atomic_json(output/(name+'.json'),dict(rule=VISIBLE_TARGET_RULE,filter=field,episodes=[r for r in rows if r[field]]))
    fields=['batch','episode_index','episode_id','seed','color','recorded_target','nearest_distance_object','object',
        'margin_mm','ambiguous','nearest_xy_matches','visible_rule_matches','unambiguous_matches','distance_0_mm','distance_1_mm']
    with (output/'episodes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for r in rows:
            values={k:r[k] for k in fields if k in r}
            values.update(margin_mm=1000*r['margin_m'],distance_0_mm=1000*r['candidates'][0]['distance_xy_m'],distance_1_mm=1000*r['candidates'][1]['distance_xy_m'])
            writer.writerow(values)
    lines=['Target rule analysis of all '+str(len(rows))+' sealed normal episodes; CPU-only, source datasets unchanged.',
        '',summary['frame']+'. '+summary['tie_policy']+'.','',
        '| Group | Episodes | Nearest XY | Visible rule | Unambiguous matches | Margin <=5 mm | Median margin mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group,s in [('ALL',summary['counts']),*summary['per_color'].items(),*summary['per_batch'].items()]:
        lines.append(f"| {group} | {s['episodes']} | {s['nearest_xy_matches']} | {s['visible_rule_matches']} | {s['unambiguous_matches']} | {s['ambiguous']} | {s['margin_mm']['median']:.3f} |")
    lines+=['','Margin distributions (mm), source hashes and exact per-episode candidate coordinates are in summary.json / episodes.json.',
        'rule-consistent.json includes the documented tie-break; unambiguous-rule-consistent.json excludes all <=5 mm layouts.',
        'Lists identify batch/dataset, local episode index, episode id and seed. They are candidates for a training subset, not a training launch.',
        'Wall time: '+str(round(summary['wall_seconds'],3))+' s.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=confined(a.output)
    if not output.is_relative_to(CANONICAL_ROOT/'analysis/target-rule'):raise ValueError('Analysis output must stay under analysis/target-rule')
    summary,rows=analyze(CANONICAL_ROOT/'datasets');publish(output,summary,rows)
    print(json.dumps(dict(passed=True,output=str(output),counts=summary['counts'],per_color=summary['per_color'],wall_seconds=summary['wall_seconds'])))


if __name__=='__main__':main()
