#!/usr/bin/env python3
"""Export audited successful attempts through the installed LeRobotDataset writer and validate."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from contract import JOINTS, timing_audit
from recovery import receipt_error, recovery_summary
from storage import confined, prepare_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--repo-id', default='local/soarm101_headless_pilot')
    p.add_argument('--report-dir', type=Path, required=True)
    args = p.parse_args()
    run, report = args.run, args.report_dir
    run, report = confined(run), confined(report)
    if (run / 'superseded.json').exists():
        raise ValueError('Superseded diagnostic run cannot be exported as a dataset')
    summary = json.loads((run / 'summary.json').read_text())
    if not summary.get('complete') or summary.get('end_marker') != 'COLLECTION_FINISHED':
        raise ValueError('Collection is incomplete; refusing to export a partial run')
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'run.json').read_text())
    prepare_environment(config['instance'])
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
    dataset_root = run / 'dataset'
    dataset = LeRobotDataset.create(args.repo_id, fps=fps, root=dataset_root,
        features=features, robot_type='so_follower', vcodec='h264', encoder_threads=1,
        image_writer_threads=2, tolerance_s=.001)
    audits, manifest, thumbs = [], [], []
    accepted_paths = []
    for path in sorted(run.glob('attempt-*')):
        receipt = json.loads((path / 'receipt.json').read_text())
        if not receipt['accepted']:
            continue
        recovery_error=receipt_error(receipt)
        if recovery_error:raise ValueError(f'{path}: {recovery_error}')
        rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
        audit = timing_audit(rows, fps)
        if not audit['passed']:
            raise ValueError(f'{path}: {audit}')
        captures = {key: cv2.VideoCapture(str(path / f'{key}.mp4')) for key in ('wrist', 'top')}
        first_stamp = rows[0]['sim_ns']
        for index, row in enumerate(rows):
            frame = {'observation.state': np.rad2deg(np.array(row['state'], dtype=np.float32)),
                     'action': np.rad2deg(np.array(row['action'], dtype=np.float32)),
                     'task': receipt['params']['task']}
            for key, capture in captures.items():
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f'{path}: {key} has fewer video frames than audit rows')
                frame[f'observation.images.{key}'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if key == 'top' and index == len(rows) // 2:
                    thumb = Image.fromarray(frame[f'observation.images.{key}']).resize((256, 192))
                    ImageDraw.Draw(thumb).text((4, 4), f'{len(manifest):02d} {receipt["params"]["color"]} seed {receipt["params"]["seed"]}', fill='white', stroke_fill='black', stroke_width=1)
                    thumbs.append(thumb)
            dataset.add_frame(frame)
            # This pinned LeRobot writer generates index/fps internally. Replace that entry with
            # the measured sim-stamp delta; the source JSONL remains the independent clock authority.
            dataset.writer.episode_buffer['timestamp'][-1] = (row['sim_ns'] - first_stamp) / 1e9
        for capture in captures.values():
            if capture.read()[0]:
                raise ValueError(f'{path}: excess video frames')
            capture.release()
        dataset.save_episode(parallel_encoding=False)
        audits.append(audit)
        manifest.append(dict(episode_index=len(manifest), **receipt))
        accepted_paths.append(path)
    dataset.finalize()
    recovery=recovery_summary(manifest)
    (dataset_root/'meta/collection_episodes.json').write_text(json.dumps(manifest,indent=2))
    loaded = LeRobotDataset(args.repo_id, root=dataset_root, video_backend='pyav', tolerance_s=.001)
    assert loaded.fps == 30 and loaded.num_episodes == len(manifest)
    assert loaded.features['action']['names'] == names
    assert loaded.features['observation.state']['names'] == names
    first, last = loaded[0], loaded[len(loaded) - 1]
    assert tuple(first['observation.state'].shape) == (6,)
    assert tuple(first['observation.images.wrist'].shape) == (3, 480, 640)
    assert tuple(last['observation.images.top'].shape) == (3, 480, 640)
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
    validation = dict(recovery=recovery,loaded=True, fps=loaded.fps, episodes=loaded.num_episodes,
                     frames=len(loaded), features=loaded.features, timing_passed=all(a['passed'] for a in audits),
                     color_counts={color: sum(e['params']['color'] == color for e in manifest) for color in ('red', 'green', 'blue')})
    (report / 'validation.json').write_text(json.dumps(validation, indent=2))
    (report / 'timing-audit.json').write_text(json.dumps(audits, indent=2))
    (report / 'episode-params.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(run / 'summary.json', report / 'summary.json')
    shutil.copy2(run / 'episodes.jsonl', report / 'attempt-log.jsonl')
    shutil.copy2(run / 'run.json', report / 'run.json')
    print(json.dumps(validation, default=str), flush=True)


if __name__ == '__main__':
    main()
