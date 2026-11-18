"""Audit stopped rule431 attempts without hiding invalid outcomes."""
import argparse,csv,json,math,statistics
from pathlib import Path

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;den=1+z*z/n;mid=(k/n+z*z/(2*n))/den
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]

def summarize(rows):
    valid=[r for r in rows if r.get('valid') is True]
    fields=('requested_colour_success','requested_colour_grasp','rule_target_success','rule_target_grasp','wrong_color_lift','wrong_same_colour_target_grasp')
    result=dict(attempts=len(rows),valid=len(valid),invalid=len(rows)-len(valid),closure_phase='CONFIRMED first measured >=2deg close')
    for key in fields:
        if any(key not in r for r in valid):raise ValueError('Missing rule431 result field '+key)
        k=sum(bool(r[key]) for r in valid)
        result[key]=dict(count=k,denominator=len(valid),wilson95=wilson(k,len(valid)))
    geometry=[r['rule_closure_confirmed_geometry'] for r in valid if r.get('rule_closure_confirmed_geometry') and r['rule_closure_confirmed_geometry'].get('status')=='available']
    for key in ('xy_mm','yaw_deg','jaw_y_tilt_deg','block_x_tilt_deg'):
        values=[abs(g[key]) if key=='yaw_deg' else g[key] for g in geometry if g.get(key) is not None]
        result[key]=dict(n=len(values),mean=statistics.mean(values) if values else None)
    ambiguous=[r for r in valid if (r.get('rule_target_selection') or {}).get('ambiguous') is True]
    unambiguous=[r for r in valid if (r.get('rule_target_selection') or {}).get('ambiguous') is False]
    unknown=[r for r in valid if (r.get('rule_target_selection') or {}).get('ambiguous') not in (True,False)]
    result['ambiguous_rule_targets']=len(ambiguous);result['unknown_rule_ambiguity']=len(unknown)
    k=sum(bool(r['rule_target_success']) for r in unambiguous)
    result['rule_target_success_excluding_ambiguous']=dict(count=k,denominator=len(unambiguous),wilson95=wilson(k,len(unambiguous)))
    result['per_colour']={c:dict(valid=sum(r['color']==c for r in valid),rule_success=sum(r['color']==c and r['rule_target_success'] for r in valid),requested_success=sum(r['color']==c and r['requested_colour_success'] for r in valid)) for c in ('red','green','blue')}
    return result

def audit(root,out):
    from native_bank import reset_protocol
    ledger=json.loads((root/'window-ledger.json').read_text())
    if ledger.get('suite')=='triplet-heldout-20x3' and ledger.get('scored') is not True:raise ValueError('Triplet window lacks an explicit scored=True admission')
    if ledger.get('scored') is False:raise ValueError('Unscored windows cannot be reported as scored results')
    if not ledger.get('stopped_utc') or ledger.get('cleanup_exit_code')!=0:raise ValueError('Window must be stopped with successful cleanup')
    episodes=[];rows=[];bank_rows={}
    if ledger.get('suite') in ('heldout-bank-v2','triplet-heldout-20x3'):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collect'))
        from native_bank import load_bank
        bank_rows={r['seed']:r for r in load_bank(Path(ledger['bank']),ledger['bank_sha256'])[0]}
    for path in sorted(root.glob('*/ep*/result.json')):
        r=json.loads(path.read_text());episodes.append(r)
        if r.get('valid') is True:
            matched=r.get('matched_reset') or {}
            expected=reset_protocol(ledger['suite'])
            labels=[v for v in (r.get('reset_protocol'),matched.get('protocol')) if v is not None]
            if not labels or any(v!=expected for v in labels):raise ValueError('Missing or conflicting reset protocol in '+str(path))
            budgets=[v for v in (r.get('readiness_budget_s'),matched.get('readiness_budget_s')) if v is not None]
            if not budgets or any(v!=ledger.get('readiness_budget_s',60) for v in budgets):raise ValueError('Missing or conflicting readiness budget in '+str(path))
        g=r.get('rule_closure_confirmed_geometry') or {}
        row={k:r.get(k) for k in ('seed','color','valid','outcome','requested_colour_success','requested_colour_grasp','rule_target_success','rule_target_grasp','wrong_color_lift','wrong_same_colour_target_grasp','target_wrong_cup','wrong_block_placement','target_any_placement','rule_target_name','paired_flow_repeat','effective_action_fps','wall_time_s','error')}
        for prefix,success,grasp in (('outcome','requested_colour_success','requested_colour_grasp'),('rule_outcome','rule_target_success','rule_target_grasp')):
            row[prefix]='invalid' if not r.get('valid') else 'success' if r.get(success) else 'grasped-not-placed' if r.get(grasp) else 'no-grasp'
        row.update({k:r.get(k) for k in ('time_to_grasp_s','time_to_place_s','time_to_rule_grasp_s','time_to_rule_place_s','rtc_enabled','sim_time_s','missed_sim_slots','empty_sim_ticks','sim_dispatch_jitter_p95_ms','host_load_start','host_load_end')})
        client=r.get('client') or {};wrist=r.get('wrist_preclosure') or {};video=r.get('video') or {}
        row.update({k:client.get(k) for k in ('observe_every_actions','actions_per_chunk','chunk_size_threshold','queue_aggregation')})
        waits=[q['wall_s']*1000 for q in client.get('requests',[]) if 'wall_s' in q]
        row.update(flow_noise=json.dumps(client.get('flow_noise'),sort_keys=True),inference_wall_mean_ms=statistics.mean(waits) if waits else None,video_path=video.get('output_path'),wrist_manifest=wrist.get('manifest'))
        if ledger.get('suite')=='triplet-heldout-20x3':
            b=bank_rows[r['seed']]
            row.update(case_id=b['case_id'],layout_seed=b['layout_seed'],requested_layout_seed=b['requested_layout_seed'],sealed_intended_target=b['recorded_target'])
        proof=r.get('rule_target_selection') or {}
        diagnostic=dict(live_choice=proof.get('object'),live_candidates=proof.get('candidates'),live_margin_m=proof.get('margin_m'),ambiguous=proof.get('ambiguous'))
        if r.get('valid') and bank_rows:
            from rule_target import select_target
            b=bank_rows[r['seed']]
            sealed=select_target(json.loads(Path(b['source_receipt']).read_text())['initial_physical'],r['color'])
            diagnostic.update(bank_choice=sealed['object'],bank_candidates=sealed['candidates'],bank_margin_m=sealed['margin_m'],live_matches_bank=proof['object']==sealed['object'],cross_check_semantics='diagnostic only; live fixed target retained, not a gate')
        row.update(readiness_budget_s=ledger.get('readiness_budget_s',60),reset_protocol=r.get('reset_protocol') or (r.get('matched_reset') or {}).get('protocol'),rule_target_selection=json.dumps(diagnostic,sort_keys=True),ambiguous=proof.get('ambiguous'),path=str(path),xy_mm=g.get('xy_mm'),yaw_deg=g.get('yaw_deg'),dz_mm=g.get('dz_mm'),closure_phase=g.get('phase'))
        rows.append(row)
    if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Do not silently combine repeated attempts in one window')
    coverage=None
    if ledger.get('suite')=='triplet-heldout-20x3':
        selected=ledger.get('selected_seeds')
        offset=ledger.get('bank_offset');fixed=ledger.get('fixed_chunk_count')
        if offset not in (0,15,30,45) or fixed!=15:raise ValueError('Triplet report requires a fixed fifteen-case chunk')
        cohort=list(bank_rows)[offset:offset+15]
        if not selected or len(set(selected))!=len(selected) or [seed for seed in cohort if seed in selected]!=selected:raise ValueError('Triplet report requires an ordered subset of the fixed case cohort')
        observed=[r['seed'] for r in rows]
        if not set(observed)<=set(selected):raise ValueError('Triplet result lies outside the selected cohort')
        coverage=dict(status='complete' if set(observed)==set(cohort) else 'partial',selected_seeds=selected,fixed_chunk_seeds=cohort,attempted=observed,missing=[seed for seed in cohort if seed not in observed],missing_selected=[seed for seed in selected if seed not in observed],invalid=sum(not r.get('valid') for r in rows))
        for raw,row in zip(episodes,rows):
            if raw.get('bank_sha256')!=ledger['bank_sha256']:raise ValueError('Triplet result bank differs from its ledger')
            planned=ledger.get('paired_flow_repeat')
            if planned is None:raise ValueError('Triplet report requires declared paired noise repeat')
            if raw.get('valid'):
                from flow_noise import episode_identity
                if raw.get('paired_flow_repeat')!=planned or (raw.get('client') or {}).get('flow_noise')!=episode_identity(row['layout_seed'],planned):raise ValueError('Triplet actual noise identity differs from its layout/repeat')
            row.update(weights_sha256=raw.get('weights_sha256'),bank_sha256=ledger['bank_sha256'],paired_flow_repeat=planned)
    out.mkdir(parents=True,exist_ok=False)
    summary=summarize(episodes);summary.update(window=str(root),status=ledger['status'],all_attempts_included=True,readiness_budget_s=ledger.get('readiness_budget_s',60),reset_protocol_by_suite=(reset_protocol(ledger['suite']) if ledger.get('suite') else None))
    if coverage is not None:summary['coverage']=coverage
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if rows:
        with (out/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary))
    return summary,rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.root,a.out)
