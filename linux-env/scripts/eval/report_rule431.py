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
    result['per_colour']={c:dict(valid=sum(r['color']==c for r in valid),rule_success=sum(r['color']==c and r['rule_target_success'] for r in valid),requested_success=sum(r['color']==c and r['requested_colour_success'] for r in valid)) for c in ('red','green','blue')}
    return result

def audit(root,out):
    ledger=json.loads((root/'window-ledger.json').read_text())
    if not ledger.get('stopped_utc') or ledger.get('cleanup_exit_code')!=0:raise ValueError('Window must be stopped with successful cleanup')
    episodes=[];rows=[]
    for path in sorted(root.glob('*/ep*/result.json')):
        r=json.loads(path.read_text());episodes.append(r)
        g=r.get('rule_closure_confirmed_geometry') or {}
        row={k:r.get(k) for k in ('seed','color','valid','outcome','requested_colour_success','requested_colour_grasp','rule_target_success','rule_target_grasp','wrong_color_lift','wrong_same_colour_target_grasp','rule_target_name','paired_flow_repeat','effective_action_fps','wall_time_s','error')}
        row.update(path=str(path),xy_mm=g.get('xy_mm'),yaw_deg=g.get('yaw_deg'),dz_mm=g.get('dz_mm'),closure_phase=g.get('phase'))
        rows.append(row)
    if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Do not silently combine repeated attempts in one window')
    out.mkdir(parents=True,exist_ok=False)
    summary=summarize(episodes);summary.update(window=str(root),status=ledger['status'],all_attempts_included=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if rows:
        with (out/'episodes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.root,a.out)
