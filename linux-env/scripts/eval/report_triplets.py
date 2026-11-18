"""Audit triplet chunks, expose missing cases and compare paired layout clusters."""
import argparse
import contextlib
import csv
import datetime
import io
import json
import math
from pathlib import Path
import random
import statistics

from native_bank import load_bank
from report_rule431 import audit, wilson
from triplet_bank import COLORS, SUITE

METRICS=('requested_colour_success','requested_colour_grasp','rule_target_success','rule_target_grasp','wrong_color_lift','wrong_same_colour_target_grasp','target_wrong_cup')


def rates(rows):
    return {key:dict(count=sum(bool(r.get('valid') and r.get(key)) for r in rows),
                     denominator=len(rows),wilson95=wilson(sum(bool(r.get('valid') and r.get(key)) for r in rows),len(rows)))
            for key in METRICS}


def geometry(rows):
    result={}
    for key in ('xy_mm','yaw_deg','dz_mm'):
        values=[float(r[key]) for r in rows if r.get('valid') and isinstance(r.get(key),(int,float)) and math.isfinite(r[key])]
        if key=='yaw_deg':values=list(map(abs,values))
        result[key]=dict(n=len(values),mean=statistics.mean(values) if values else None,median=statistics.median(values) if values else None)
    result['phase']='CONFIRMED >=2deg; yaw is absolute horizontal relative-axis error; unavailable closures excluded with count shown'
    return result


def aggregate(windows,out):
    out=Path(out)
    if out.exists():raise FileExistsError('Report output already exists')
    windows=[Path(root).resolve() for root in windows]
    if len(set(windows))!=len(windows):raise ValueError('Duplicate window path')
    manifests=[json.loads((root/'window-ledger.json').read_text()) for root in windows]
    for ledger in manifests:
        datetime.datetime.strptime(ledger['started_utc'],'%Y-%m-%dT%H:%M:%SZ')
    ordered=sorted(zip(windows,manifests),key=lambda item:item[1]['started_utc'])
    if not manifests or any(r.get('suite')!=SUITE for r in manifests):raise ValueError('Only scored triplet windows can be aggregated')
    banks={(r['bank'],r['bank_sha256']) for r in manifests}
    if len(banks)!=1:raise ValueError('Cannot mix banks')
    root,bank_sha=next(iter(banks));bank_rows,_=load_bank(root,bank_sha)
    expected={r['seed']:r for r in bank_rows}
    out.mkdir(parents=True)
    all_rows=[];chunk_reports=[]
    for index,(window,ledger) in enumerate(ordered):
        with contextlib.redirect_stdout(io.StringIO()):summary,rows=audit(Path(window),out/f'chunk-{index:02d}')
        for row in rows:row['window_started_utc']=ledger['started_utc']
        all_rows.extend(rows);chunk_reports.append(summary)
    weights={r['weights_sha256'] for r in all_rows}
    if len(weights)>1 or (all_rows and (None in weights or any(len(v)!=64 for v in weights))):raise ValueError('Cannot mix or omit checkpoint weights')
    repeats={r['paired_flow_repeat'] for r in all_rows}
    if len(repeats)>1 or (all_rows and None in repeats):raise ValueError('Cannot mix inference repeats across chunks')
    grouped={}
    for row in all_rows:
        if row['seed'] not in expected:raise ValueError('Unregistered triplet case')
        grouped.setdefault(row['seed'],[]).append(row)
    primary=[]
    for seed,attempts in grouped.items():
        starts=[r['window_started_utc'] for r in attempts]
        if len(set(starts))!=len(starts):raise ValueError('Attempt chronology is ambiguous')
        primary.append(min(attempts,key=lambda row:row['window_started_utc']))
    primary.sort(key=lambda r:list(expected).index(r['seed']))
    missing=[r['case_id'] for seed,r in expected.items() if seed not in grouped]
    report=dict(kind='triplet360-audited-results-v1',bank_sha256=bank_sha,weights_sha256=next(iter(weights)) if weights else None,
        coverage=dict(status='partial' if missing else 'complete',expected_cases=60,expected_layouts=20,
                      primary_attempts=len(primary),all_attempts=len(all_rows),invalid_attempts=sum(not r['valid'] for r in all_rows),
                      missing_cases=missing,missing_case_keys=[seed for seed in expected if seed not in grouped],missing_by_offset={str(offset):[row['seed'] for row in bank_rows[offset:offset+15] if row['seed'] not in grouped] for offset in (0,15,30,45)},repeated_cases=[expected[s]['case_id'] for s,v in grouped.items() if len(v)>1]),
        outcome_semantics='First chronological attempt per case is immutable primary, including invalid failures. Later attempts never replace earlier outcomes; they may fill only previously unattempted cases. All attempts retained.',
        metrics=rates(primary),valid_only_metrics=rates([r for r in primary if r['valid']]),closure_geometry=geometry(primary),
        per_colour={c:dict(attempted=sum(r['color']==c for r in primary),invalid=sum(r['color']==c and not r['valid'] for r in primary),metrics=rates([r for r in primary if r['color']==c])) for c in COLORS},
        uncertainty='Wilson intervals are descriptive by case; RGB outcomes within one layout are dependent. Use layout-cluster pairing for checkpoint differences. Small-N bootstrap degeneracy does not establish equality.',
        primary_rows=primary,all_rows=all_rows,chunk_reports=chunk_reports)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if all_rows:
        with (out/'all-attempts.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    return report


def paired(left,right,draws=10000):
    if left['bank_sha256']!=right['bank_sha256']:raise ValueError('Paired reports require the same bank')
    def layouts(report):
        result={}
        for row in report['primary_rows']:result.setdefault(row['layout_seed'],{})[row['color']]=row
        return {k:v for k,v in result.items() if set(v)==set(COLORS)}
    a,b=layouts(left),layouts(right);shared=sorted(set(a)&set(b))
    if not shared:raise ValueError('No shared complete RGB layout clusters')
    if any(a[seed][color].get('paired_flow_repeat') is None or a[seed][color].get('paired_flow_repeat')!=b[seed][color].get('paired_flow_repeat') for seed in shared for color in COLORS):raise ValueError('Paired reports require identical declared inference repeats')
    rng=random.Random(360431)
    result=dict(shared_layouts=shared,shared_cases=3*len(shared),paired_unit='layout; all three colours resampled together',draws=draws,differences_right_minus_left={})
    for key in METRICS:
        values=[]
        for seed in shared:
            values.append(statistics.mean(float(bool(b[seed][c]['valid'] and b[seed][c].get(key)))-float(bool(a[seed][c]['valid'] and a[seed][c].get(key))) for c in COLORS))
        samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(draws))
        result['differences_right_minus_left'][key]=dict(mean=statistics.mean(values),layout_cluster_percentile95=[samples[int(.025*(draws-1))],samples[int(.975*(draws-1))]])
    for key in ('xy_mm','yaw_deg','dz_mm'):
        clusters=[]
        for seed in shared:
            values=[]
            for color in COLORS:
                x,y=a[seed][color],b[seed][color]
                if x['valid'] and y['valid'] and all(isinstance(row.get(key),(int,float)) and math.isfinite(row[key]) for row in (x,y)):
                    values.append((abs(y[key])-abs(x[key])) if key=='yaw_deg' else y[key]-x[key])
            if values:clusters.append(values)
        if clusters:
            samples=[]
            for _ in range(draws):
                sample=[v for cluster in rng.choices(clusters,k=len(clusters)) for v in cluster]
                samples.append(statistics.mean(sample))
            samples.sort()
            result['differences_right_minus_left'][key]=dict(mean=statistics.mean(v for cluster in clusters for v in cluster),paired_closures=sum(map(len,clusters)),layout_clusters=len(clusters),layout_cluster_percentile95=[samples[int(.025*(draws-1))],samples[int(.975*(draws-1))]])
    result['limitation']='Empirical paired uncertainty on observed layouts/noise only. A zero-width interval when all pairs tie is not proof of no effect; there are only20 held layouts and one inference repeat.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('windows',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    a=p.parse_args();report=aggregate(a.windows,a.out)
    if a.compare:
        comparison=paired(json.loads(a.compare.read_text()),report)
        (a.out/'paired-differences.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(dict(coverage=report['coverage'],metrics=report['metrics'])))


if __name__=='__main__':main()
