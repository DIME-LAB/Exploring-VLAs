"""Run the unchanged ROS exporter, then check every numeric LeRobot frame."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import shared
from storage import confined,atomic_json,prepare_environment
from collection_status import require_finalized
from contract import timing_audit
from recovery import receipt_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--repo-id',default='local/soarm101_inproc_v1')
    a=p.parse_args();root=confined(a.run);report=confined(a.report_dir)
    require_finalized(root)
    config=json.loads((root/'run.json').read_text())
    prepare_environment(config['instance'])
    if config.get('physics_only'):raise ValueError('Deferred-render candidates have not passed the live/replay fidelity gate')
    subprocess.run([sys.executable,str(shared.REPO/'linux-env/scripts/collect/export.py'),str(root),
        '--repo-id',a.repo_id,'--report-dir',str(report)],check=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset=LeRobotDataset(a.repo_id,root=root/'dataset',video_backend='pyav',tolerance_s=.001)
    index=0;episodes=0
    for path in sorted(root.glob('attempt-*')):
        receipt=json.loads((path/'receipt.json').read_text())
        if not receipt.get('accepted'):continue
        rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
        if receipt_error(receipt,rows) or not timing_audit(rows,30)['passed']:raise ValueError('Raw audit failed')
        if any(r.get('render_pending') for r in rows):raise ValueError('Images have not been captured live')
        for row in rows:
            frame=dataset.hf_dataset[index]
            for feature,raw in (('observation.state','state'),('action','action')):
                expected=np.rad2deg(np.asarray(row[raw],np.float32))
                if not np.allclose(np.asarray(frame[feature]),expected,atol=1e-5,rtol=0):
                    raise ValueError('Degree feature differs from raw frame: '+feature+' index '+str(index))
            expected_time=(row['sim_ns']-rows[0]['sim_ns'])/1e9
            if abs(float(frame['timestamp'])-expected_time)>1e-5:raise ValueError('Dataset timestamp differs')
            if int(frame['episode_index'])!=episodes:raise ValueError('Dataset episode index differs')
            index+=1
        episodes+=1
    if index!=len(dataset) or episodes!=dataset.num_episodes:raise ValueError('Export count differs')
    atomic_json(report/'full-readback.json',dict(passed=True,frames=index,episodes=episodes,
        measured_state_degrees=True,clean_action_degrees=True,sim_timestamps=True,
        exporter='linux-env/scripts/collect/export.py'))


if __name__=='__main__':main()
