#!/usr/bin/env python3
"""Export audited successful attempts through the installed LeRobotDataset writer and validate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Audit shares the calibrated pure IK package without importing the ROS node.
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'vla_SO-ARM101/src/so_arm101_control'))

from contract import JOINTS, timing_audit
from tick_contract import camera_tick_errors
from collection_status import require_finalized
from checkpoint import require_recovery_mode,require_episode_mode
from recovery import receipt_error, recovery_summary
from dataset_phase import FEATURE,PHASES,POLICY as PHASE_POLICY,phase_code,feature,stored_phase_codes
from storage import atomic_json,confined, prepare_environment
from export_content import validate_metadata
from export_video import validate_videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--repo-id', default='local/soarm101_headless_pilot')
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--validate-existing',action='store_true',help='Read back a finalized dataset and verify it against raw receipts/frames without rewriting it')
    args = p.parse_args()
    run, report = args.run, args.report_dir
    run, report = confined(run), confined(report)
    report.mkdir(parents=True, exist_ok=True)
    validation_started=time.time_ns()
    atomic_json(report/'validation.json',dict(passed=False,loaded=False,status='VALIDATION_IN_PROGRESS',started_wall_ns=validation_started))
    if (run / 'superseded.json').exists():
        raise ValueError('Superseded diagnostic run cannot be exported as a dataset')
    require_finalized(run)
    summary = json.loads((run / 'summary.json').read_text())
    if not summary.get('complete') or summary.get('end_marker') != 'COLLECTION_FINISHED':
        raise ValueError('Collection is incomplete; refusing to export a partial run')
    config = json.loads((run / 'run.json').read_text())
    require_recovery_mode(config,config.get('regrasp_only',False),config.get('normal_only',False))
    for path in sorted(run.glob('attempt-*/receipt.json')):
        receipt=json.loads(path.read_text())
        require_episode_mode(receipt['params'],config.get('regrasp_only',False),config.get('normal_only',False))
        if config.get('regrasp_policy',{}).get('strategy')=='synthetic_post_miss_reset' and receipt['params'].get('recovery')=='regrasp' and receipt.get('accepted'):
            if receipt.get('recovery_report',{}).get('strategy')!='synthetic_post_miss_reset':raise ValueError('Current recovery policy requires a certified synthetic post-miss reset')
    prepare_environment(config['instance'])
    validator_names=('export.py','export_content.py','export_video.py','dataset_phase.py')
    validator_sources={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in validator_names}
    import cv2
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    fps = config['fps']
    from lerobot.robots.so101_ros2.config_so101_ros2 import SO101ROS2RobotConfig
    deployment = SO101ROS2RobotConfig(use_degrees=True, robot_type='so_follower')
    names = [name + '.pos' for name in deployment.joint_names]
    features = {key: dict(dtype='float32', shape=(6,), names=names) for key in ('observation.state', 'action')}
    features.update({f'observation.images.{key}': dict(dtype='video', shape=(480, 640, 3),
                    names=['height', 'width', 'channels']) for key in ('wrist', 'top')})
    features[FEATURE]=feature()
    dataset_root = run / 'dataset'
    if args.validate_existing:
        # These files are published only after the dataset writer finalizes.
        stored_manifest=json.loads((dataset_root/'meta/collection_episodes.json').read_text())
        stored_phases=json.loads((dataset_root/'meta/collection_phases.json').read_text())
        dataset=None
    else:
        dataset = LeRobotDataset.create(args.repo_id, fps=fps, root=dataset_root,
            features=features, robot_type='so_follower', vcodec='h264', encoder_threads=1,
            image_writer_threads=2, tolerance_s=.001)
    audits, manifest, thumbs = [], [], []
    phase_counts={name:0 for name in PHASES};phase_values=[];expected_states=[];expected_actions=[]
    accepted_paths = []
    for path in sorted(run.glob('attempt-*')):
        receipt = json.loads((path / 'receipt.json').read_text())
        if not receipt['accepted']:
            continue
        rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
        if config.get('camera_selection_policy',{}).get('version',0)>=3:
            selection_errors=[error for row in rows for error in camera_tick_errors(row,required=True)]
            if selection_errors:raise ValueError(f'{path}: {selection_errors}')
        recovery_error=receipt_error(receipt,rows)
        if recovery_error:raise ValueError(f'{path}: {recovery_error}')
        audit = timing_audit(rows, fps)
        if not audit['passed']:
            raise ValueError(f'{path}: {audit}')
        captures = {key: cv2.VideoCapture(str(path / f'{key}.mp4')) for key in ('wrist', 'top')}
        first_stamp = rows[0]['sim_ns']
        for index, row in enumerate(rows):
            code=phase_code(row,receipt,required='frame_phase_policy' in config)
            phase_values.append(code)
            phase_counts[next(name for name,value in PHASES.items() if value==code)]+=1
            frame = {'observation.state': np.rad2deg(np.array(row['state'], dtype=np.float32)),
                     'action': np.rad2deg(np.array(row['action'], dtype=np.float32)),
                     'task': receipt['params']['task'],FEATURE:np.asarray([code],dtype=np.int64)}
            for key, capture in captures.items():
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f'{path}: {key} has fewer video frames than audit rows')
                frame[f'observation.images.{key}'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if key == 'top' and index == len(rows) // 2:
                    thumb = Image.fromarray(frame[f'observation.images.{key}']).resize((256, 192))
                    ImageDraw.Draw(thumb).text((4, 4), f'{len(manifest):02d} {receipt["params"]["color"]} seed {receipt["params"]["seed"]}', fill='white', stroke_fill='black', stroke_width=1)
                    thumbs.append(thumb)
            expected_states.append(frame['observation.state']);expected_actions.append(frame['action'])
            if dataset is not None:
                dataset.add_frame(frame)
                # Persist the measured stamp delta rather than a generated index/fps timestamp.
                dataset.writer.episode_buffer['timestamp'][-1] = (row['sim_ns'] - first_stamp) / 1e9
        for capture in captures.values():
            if capture.read()[0]:
                raise ValueError(f'{path}: excess video frames')
            capture.release()
        if dataset is not None:dataset.save_episode(parallel_encoding=False)
        audits.append(audit)
        manifest.append(dict(episode_index=len(manifest), **receipt))
        accepted_paths.append(path)
    if dataset is not None:dataset.finalize()
    recovery=recovery_summary(manifest)
    phases=dict(PHASE_POLICY,counts=phase_counts)
    if args.validate_existing:
        if stored_manifest!=manifest or stored_phases!=phases:raise ValueError('Existing dataset manifest/phases differ from audited source')
    else:
        atomic_json(dataset_root/'meta/collection_episodes.json',manifest)
        atomic_json(dataset_root/'meta/collection_phases.json',phases)
    loaded = LeRobotDataset(args.repo_id, root=dataset_root, video_backend='pyav', tolerance_s=.001)
    assert loaded.fps == 30 and loaded.num_episodes == len(manifest)
    assert loaded.features['action']['names'] == names
    assert loaded.features['observation.state']['names'] == names
    first, last = loaded[0], loaded[len(loaded) - 1]
    assert tuple(loaded.features[FEATURE]['shape'])==(1,) and loaded.features[FEATURE]['dtype']=='int64'
    # LeRobot/HF may expose a one-element int64 column as a scalar tensor.
    assert first[FEATURE].numel()==1 and first[FEATURE].dtype==torch.int64
    assert stored_phase_codes(loaded)==phase_values
    for key,expected in (('observation.state',expected_states),('action',expected_actions)):
        actual=np.asarray([value.tolist() for value in loaded.hf_dataset[key]],dtype=np.float32)
        if actual.shape!=np.asarray(expected).shape or not np.allclose(actual,expected,atol=5e-5,rtol=0.):
            raise ValueError('Stored '+key+' differs from audited source')
    assert tuple(first['observation.state'].shape) == (6,)
    assert tuple(first['observation.images.wrist'].shape) == (3, 480, 640)
    assert tuple(last['observation.images.top'].shape) == (3, 480, 640)
    try:
        content=validate_metadata(run)
        video_content=validate_videos(run,content,sample_count=None)
        if validator_sources!={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in validator_names}:
            raise RuntimeError('Validator source changed during content audit')
    except Exception as error:
        atomic_json(report/'validation.json',dict(passed=False,loaded=False,status='VALIDATION_FAILED',
            started_wall_ns=validation_started,completed_wall_ns=time.time_ns(),error=str(error),
            field=getattr(error,'field',None),details=getattr(error,'details',None),validator_source_sha256=validator_sources))
        raise
    sheet = Image.new('RGB', (256 * 5, 192 * ((len(thumbs) + 4) // 5)))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 5) * 256, (index // 5) * 192))
    sheet.save(report / 'contact-sheet.jpg')
    colors = set()
    for path, ep in sorted(zip(accepted_paths, manifest),key=lambda pair:pair[1]['params'].get('recovery')!='regrasp'):
        color = ep['params']['color']
        if color in colors:
            continue
        colors.add(color)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path / 'top.mp4'), '-i', str(path / 'wrist.mp4'),
            '-filter_complex', '[0:v][1:v]hstack=inputs=2', '-c:v', 'libx264', '-threads', '2', '-crf', '20',
            str(report / f'sample-{color}.mp4')], check=True)
    validation = dict(passed=True,status='VALIDATION_FINISHED',started_wall_ns=validation_started,completed_wall_ns=time.time_ns(),
                     validator_source_sha256=validator_sources,
                     export_content=dict(metadata=content,video=video_content),
                     frame_phases=dict(policy=PHASE_POLICY,counts=phase_counts,all_values_read_back=True),recovery=recovery,loaded=True, fps=loaded.fps, episodes=loaded.num_episodes,
                     frames=len(loaded), features=loaded.features,all_state_actions_read_back=True,timing_passed=all(a['passed'] for a in audits),
                     color_counts={color: sum(e['params']['color'] == color for e in manifest) for color in ('red', 'green', 'blue')})
    atomic_json(report / 'validation.json',validation)
    atomic_json(report / 'timing-audit.json',audits)
    atomic_json(report / 'episode-params.json',manifest)
    atomic_json(report/'summary.json',json.loads((run/'summary.json').read_text()))
    shutil.copy2(run / 'episodes.jsonl', report / 'attempt-log.jsonl')
    atomic_json(report/'run.json',json.loads((run/'run.json').read_text()))
    print(json.dumps(validation, default=str), flush=True)


if __name__ == '__main__':
    main()
