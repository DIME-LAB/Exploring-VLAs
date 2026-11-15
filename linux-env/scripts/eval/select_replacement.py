"""Freeze a replacement setting from the completed 15-episode diagnostic."""
from pathlib import Path
import argparse,hashlib,json,math

def select(root):
    root=Path(root);ledger=json.loads((root/'window-ledger.json').read_text())
    assert ledger['status']=='matrix-finished' and ledger['cleanup_exit_code']==0 and ledger['stopped_utc']
    rows=[];files={}
    for cadence in (10,25,50):
        paths=sorted((root/f'repl{cadence}').glob('ep*/result.json'));assert len(paths)==5
        episodes=[json.loads(p.read_text()) for p in paths]
        assert sorted(r['seed'] for r in episodes)==list(range(530000,530005))
        assert all(r['valid'] and not r['rtc_enabled'] and r['queue_aggregation']=='latest_only' and r['client']['observe_every_actions']==cadence for r in episodes)
        distances=[r['closing_distance_m']*1000 for r in episodes if r['closing_distance_m'] is not None]
        rows.append(dict(cadence=cadence,placements=sum(r['success'] for r in episodes),target_grasps=sum(r['grasp_success'] for r in episodes),missing_closures=5-len(distances),mean_first_close_mm=sum(distances)/len(distances) if distances else None,mean_closest_mm=sum(r['closest_target_distance_m']*1000 for r in episodes)/5))
        files.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    def rank(row):return (-row['placements'],-row['target_grasps'],row['missing_closures'],row['mean_first_close_mm'] if row['mean_first_close_mm'] is not None else math.inf,row['mean_closest_mm'],row['cadence'])
    winner=min(rows,key=rank)
    return dict(complete=True,selected_cadence=winner['cadence'],criterion='Most placements, then target grasps, then fewest missing closures, then lowest mean first-closing distance, then closest distance, then lower cadence; frozen before memorisation evaluation',conditions=rows,source_window=str(root.resolve()),source_ledger_sha256=hashlib.sha256((root/'window-ledger.json').read_bytes()).hexdigest(),selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files_sha256=files)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();decision=select(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
