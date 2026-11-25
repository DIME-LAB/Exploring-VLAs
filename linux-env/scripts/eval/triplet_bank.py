"""Seal recorded triplet homes for native evaluation and prompt counterfactuals.

The registry, not a numeric seed range, defines membership. Videos remain in
their SHA-bound source directories; only one decoded home pair per layout is
materialized. No simulator, ROS or model is imported here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import shutil
import sys

HELD_SHA = 'ae7165f2619de58b3bb5f201980fa20f0e71cabe961394fb1736cf04a72cca61'
TRAIN_SHA = '05f93c2b901b6880e536dada762addaa30f2584775bd9ee7cd34b6a0bd1fb14d'
PROJECTION_SHA = '80609440684caf57eeced360f714c3e7919d66424457b28d2231691e7cc44051'
COLLECTOR_CUT = '503e39a692d47056ea325b7a20baa2f1d6900c91'
COLORS = ('red', 'green', 'blue')
SUITE = 'triplet-heldout-20x3'
RESET = 'collector-recorded-triplet-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def document(path, expected=None):
    path = Path(path)
    if expected is not None and sha(path) != expected:
        raise ValueError('SHA mismatch: ' + str(path))
    return json.loads(path.read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def registries(held, train, projection):
    h = document(held, HELD_SHA)
    t = document(train, TRAIN_SHA)
    p = document(projection, PROJECTION_SHA)
    for registry, count in ((h, 20), (t, 120)):
        if registry['sealed'] is not True or registry['cut'] != COLLECTOR_CUT or len(registry['layouts']) != count:
            raise ValueError('Registry cut, completion or cohort differs')
        actual = [row['layout_seed'] for row in registry['layouts']]
        if len(set(actual)) != count:
            raise ValueError('Duplicate actual layout')
    if set(r['layout_seed'] for r in h['layouts']) & set(r['layout_seed'] for r in t['layouts']):
        raise ValueError('Train/held layout overlap')
    if p['authoritative_registry_sha256'] != HELD_SHA or len(p['layouts']) != 20:
        raise ValueError('Projection authority differs')
    for a, b in zip(h['layouts'], p['layouts']):
        for key in a:
            if key not in ('raw_root', 'dataset_root', 'validation_path') and a[key] != b[key]:
                raise ValueError('Projection membership differs: ' + key)
        for key in ('raw_root', 'dataset_root', 'validation_path'):
            suffix = a[key].split('/datasets/counterfactual-targets/', 1)[1]
            if not b[key].endswith('/datasets/counterfactual-targets/' + suffix):
                raise ValueError('Projection source path differs')
    return h, t, p


def first_rgb(path):
    import numpy as np
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
                                   '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                   '-threads', '1', 'pipe:1'])
    if len(raw) != 480 * 640 * 3:
        raise ValueError('Home camera is not RGB 640x480')
    return np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3).copy()


def verify_sources(entry, collector_folder):
    """Reuse the producer's sealed group validator, including native frame joins."""
    root = Path(entry['raw_root']).resolve()
    for name, digest in entry['raw_files_sha256'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError('Recorded member file changed: ' + str(path))
    validation = document(entry['validation_path'], entry['validation_sha256'])
    if validation['passed'] is not True:
        raise ValueError('Source export validation did not pass')
    sys.path.insert(0, str(collector_folder))
    from target_triplets import validate_run
    # The dual projection preserves the producer's canonical relative paths.
    base = root.parents[3]
    certified = validate_run(root, base=base)
    if certified != validation['counterfactual_triplet']:
        raise ValueError('Recomputed triplet differs from registry-bound validation')
    return root, certified


def source_rows(entry, root, certified):
    from comparisons import matches
    group = []
    for attempt in sorted(root.glob('attempt-*/receipt.json')):
        receipt = document(attempt)
        color = receipt['params']['color']
        native_path = attempt.with_name('first-recorded-native.json')
        native = document(native_path)['native']
        frames = attempt.with_name('frames.jsonl')
        with frames.open() as stream:
            first = json.loads(stream.readline())
        if not matches(first['sim_ns'], native['sim_ns'], 'native_ns') or first['epoch'] != native['epoch'] or first['physics_state'] != native['state']:
            raise ValueError('First camera/state/native tick join differs')
        if color not in COLORS or receipt['params']['seed'] != entry['layout_seed']:
            raise ValueError('Wrong member color/layout')
        group.append(dict(
            seed=entry['layout_seed'] * 10 + COLORS.index(color),
            case_id=f"{entry['layout_seed']}:{color}", layout_seed=entry['layout_seed'],
            requested_layout_seed=entry['requested_layout_seed'], color=color,
            task=receipt['params']['task'], recorded_target=receipt['params']['object'],
            state=receipt['episode_start_physical']['state'],
            recorded_drive_targets=receipt['episode_start_physical']['drive_targets'],
            robot_base_world_pose=receipt['episode_start_physical']['robot_base_world_pose'],
            base_world_pose=receipt['episode_start_physical']['base_world_pose'],
            bodies={k:{f:v[f] for f in ('position','orientation')} for k,v in receipt['episode_start_physical']['bodies'].items()},
            source_receipt=str(attempt), source_receipt_sha256=sha(attempt),
            source_frames=str(frames), source_frames_sha256=sha(frames),
            first_frame_reference=str(frames), first_native=str(native_path), first_native_sha256=sha(native_path),
            reference_run=str(root/'run.json'), reference_run_sha256=sha(root/'run.json'),
            anchor_path=str(root/'triplet-anchor.json'), anchor_sha256=sha(root/'triplet-anchor.json'),
            source_trial=receipt['params']['trial'], triplet_id=certified['triplet_id'],
            target_selection=receipt['params']['target_selection']))
    by_color = {row['color']:row for row in group}
    if len(group) != 3 or set(by_color) != set(COLORS):
        raise ValueError('Incomplete RGB group')
    return [by_color[c] for c in COLORS]


def build(held, train, projection, collector_root, out):
    import numpy as np
    out = Path(out).resolve()
    owned = Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval')
    if not out.is_relative_to(owned) or shutil.disk_usage(owned).free < 100 * 1024**3 + 64 * 1024**2:
        raise ValueError('Bank output must preserve the owned backup root and 100 GiB floor')
    if out.exists():
        raise ValueError('Refusing to overwrite a sealed bank')
    collector_root = Path(collector_root).resolve()
    actual_cut = subprocess.check_output(['git','-C',str(collector_root),'rev-parse','HEAD'],text=True).strip()
    if actual_cut != COLLECTOR_CUT:
        raise ValueError('Collector checkout differs from training')
    _, _, projected = registries(held, train, projection)
    out.mkdir(parents=True)
    rows, homes, sources = [], [], []
    for entry in projected['layouts']:
        root, certified = verify_sources(entry, collector_root/'linux-env/scripts/collect')
        group = source_rows(entry, root, certified)
        rows.extend(group)
        red = group[0]
        native = document(red['first_native'])['native']
        pixels = {k:first_rgb(Path(red['source_receipt']).with_name(k+'.mp4')) for k in ('top','wrist')}
        camera_name = f"home-{entry['layout_seed']}.npz"
        np.savez_compressed(out/camera_name, **pixels)
        homes.append(dict(seed=entry['layout_seed'], requested_layout_seed=entry['requested_layout_seed'],
            source_prompt='red', state_degrees=[math.degrees(v) for v in native['state']],
            epoch=native['epoch'], sim_ns=native['sim_ns'], robot_base_world_pose=native['robot_base_world_pose'],
            blocks={k:v for k,v in native['bodies'].items() if '_2x3_' in k},
            cameras=camera_name, pixel_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in pixels.items()},
            camera_representation='decoded original recorder libx264 CRF18 RGB; no second export encode',
            source_first_native=red['first_native'], source_first_native_sha256=red['first_native_sha256'],
            expert_by_prompt={r['color']:r for r in group}, triplet_id=certified['triplet_id']))
        # Bind additional producer files, already validated against the sealed group.
        sources.append(dict(raw_root=str(root), files_sha256={name:sha(root/name) for name in
            ('run.json','triplet-anchor.json','counterfactual-triplet.json','summary.json')}))
    write_json(out/'cases.json', rows)
    write_json(out/'held20-inputs.json', homes)
    manifest = dict(format_version=1, kind=SUITE, complete=True, collector_cut=COLLECTOR_CUT,
        reset_protocol=RESET, authoritative_held_registry=dict(path=str(Path(held).resolve()),sha256=HELD_SHA),
        authoritative_train_registry=dict(path=str(Path(train).resolve()),sha256=TRAIN_SHA),
        projection=dict(path=str(Path(projection).resolve()),sha256=PROJECTION_SHA),
        cases=len(rows), layouts=len(homes), colors=list(COLORS),
        seeds=[r['seed'] for r in rows], layout_seeds=[r['seed'] for r in homes],
        case_identity='seed is legacy integer case key = actual_layout_seed*10 + RGB index; never a sampler seed',
        noise_identity='actual layout_seed and repeat index, identical across RGB and checkpoints',
        chunk_offsets=[0,15,30,45], chunk_count=15, source_groups=sources,
        files_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    write_json(out/'bank-manifest.json', manifest)
    return dict(status='CPU_BANK_SEALED', bank=str(out), bank_sha256=sha(out/'bank-manifest.json'),
                layouts=len(homes), cases=len(rows), gpu_used=False, model_loaded=False)


def load(root, expected_sha256, verify_external=True):
    if sys.flags.optimize:raise ValueError('Triplet evaluation refuses optimized Python; assertion gates must remain active')
    root = Path(root).resolve()
    bank = document(root/'bank-manifest.json', expected_sha256)
    if not (bank['kind'] == SUITE and bank['complete'] is True and bank['cases'] == 60 and bank['layouts'] == 20 and bank['collector_cut'] == COLLECTOR_CUT):
        raise ValueError('Triplet bank contract differs')
    for name, digest in bank['files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError('Sealed bank file changed: '+name)
    if verify_external:
        registries(bank['authoritative_held_registry']['path'], bank['authoritative_train_registry']['path'], bank['projection']['path'])
        for group in bank['source_groups']:
            for name, digest in group['files_sha256'].items():
                if sha(Path(group['raw_root'])/name) != digest:
                    raise ValueError('External producer inputs changed')
    rows = document(root/'cases.json')
    if [r['seed'] for r in rows] != bank['seeds'] or len(set(bank['seeds'])) != 60:
        raise ValueError('Case ordering/identity differs')
    for i in range(0,60,3):
        group = rows[i:i+3]
        if [r['color'] for r in group] != list(COLORS) or len({r['layout_seed'] for r in group}) != 1:
            raise ValueError('RGB grouping differs')
        if group[0]['layout_seed'] != bank['layout_seeds'][i//3]:
            raise ValueError('Layout ordering differs')
        for row in group:
            if row['seed'] != row['layout_seed']*10+COLORS.index(row['color']):
                raise ValueError('Legacy case key differs')
            if verify_external:
                for key in ('source_receipt','source_frames','reference_run','first_native','anchor_path'):
                    digest_key = 'anchor_sha256' if key == 'anchor_path' else key+'_sha256'
                    if sha(row[key]) != row[digest_key]:
                        raise ValueError('Source member changed: '+key)
            row.update(bank_root=str(root),bank_sha256=expected_sha256,suite=SUITE)
    return rows, SUITE


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Bank preparation requires explicitly empty CUDA_VISIBLE_DEVICES')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--held-registry',type=Path,required=True)
    p.add_argument('--train-registry',type=Path,required=True)
    p.add_argument('--projection',type=Path,required=True)
    p.add_argument('--collector-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.held_registry,a.train_registry,a.projection,a.collector_root,a.out)))


if __name__ == '__main__':
    main()
