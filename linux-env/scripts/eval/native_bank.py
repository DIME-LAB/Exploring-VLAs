"""Read sealed native home/collector banks without opening ROS or a model."""
from pathlib import Path
import hashlib,json,math
COLORS=('red','green','blue')
NAMES={f'{c}_2x3_{i}' for c in COLORS for i in range(2)}|{f'cup_{c}' for c in COLORS}

def load_bank(root,expected_sha256):
    root=Path(root).resolve();path=root/'bank-manifest.json';raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Bank manifest SHA differs')
    bank=json.loads(raw)
    def checked(name,sha):
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Bank file SHA/path differs: '+name)
        return path
    for name,sha in bank['files_sha256'].items():checked(name,sha)
    legacy=bank.get('kind')=='seen-training-start-diagnostic'
    if legacy:
        if bank.get('mode')!='home' or bank['seeds']!=list(range(530000,530005)):raise ValueError('Unqualified seen home bank')
        if not {f'{kind}-{seed}.json' for seed in bank['seeds'] for kind in ('state','seed')}<=bank['files_sha256'].keys():raise ValueError('Missing sealed home file')
        rows=[dict(json.loads((root/f'state-{seed}.json').read_text()),bodies=json.loads((root/f'seed-{seed}.json').read_text())) for seed in bank['seeds']]
        suite='seen-training-home'
    else:
        if not (bank.get('format_version')==2 and bank.get('purpose')=='held-out-collector-scenes' and bank.get('complete') is True and bank.get('producer')=='collect.Collector.run_scene_bank' and bank['seeds']==list(range(900000,900030))):raise ValueError('Unqualified collector held-out bank')
        for name,sha in bank['sampler_metadata_sha256'].items():checked(name,sha)
        rows=[]
        for seed in bank['seeds']:
            name=f'metadata/seed-{seed}.json'
            if name not in bank['sampler_metadata_sha256'] or f'seed-{seed}.json' not in bank['files_sha256']:raise ValueError('Missing sealed scene receipt/body file')
            receipt=json.loads((root/name).read_text());params=receipt['params'];physical=receipt['initial_physical'];color=COLORS[seed%3]
            if not (receipt['accepted'] is True and receipt['scene_bank_only'] is True and params==bank['parameters'][str(seed)] and params['seed']==seed and params['color']==color and params['split']=='eval' and params['recovery']=='none' and params['task']==f'Pick a {color} lego and place it in {color} cup'):raise ValueError('Scene receipt contract differs')
            bodies=json.loads((root/f'seed-{seed}.json').read_text())
            for obj,pose in bodies.items():
                if pose!={k:physical['bodies'][obj][k] for k in ('position','orientation')}:raise ValueError('Scene bodies differ from sampler receipt')
            rows.append(dict(seed=seed,color=color,task=params['task'],state=physical['state'],recorded_drive_targets=physical['drive_targets'],robot_base_world_pose=physical['robot_base_world_pose'],bodies=bodies))
        suite='heldout-bank-v2'
    for seed,row in zip(bank['seeds'],rows):
        if row['seed']!=seed or len(row['state'])!=6 or len(row['recorded_drive_targets'])!=6 or not all(math.isfinite(v) for v in row['state']+row['recorded_drive_targets']):raise ValueError('Scene home state contract differs')
        if set(row['bodies'])!=NAMES:raise ValueError('Scene object set differs')
    return rows,suite
