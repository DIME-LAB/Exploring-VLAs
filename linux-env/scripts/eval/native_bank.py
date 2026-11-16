"""Read sealed native home/collector banks without opening ROS or a model."""
from pathlib import Path
import hashlib,json,math
COLORS=('red','green','blue')
NAMES={f'{c}_2x3_{i}' for c in COLORS for i in range(2)}|{f'cup_{c}' for c in COLORS}
COLLECTOR_SHA='09f86817b46e2b1ad4695fce29e059ea387a0bbc97d9568786a70c818308ae90'
HELPER_SHA='eb41cef2e7bee6a972e646a26963c8a0d01142895223969c880144401e4ee868'
SEEN15_SEEDS=tuple(range(530000,530006))+tuple(range(530500,530506))+tuple(range(531000,531003))
SEEN_SUITES=('seen-training-home','seen-training-home-15')
SEEN_BANK_SHA='dbae9bdc10cc5755d6aeba9a6038de3b778614572c6950afa5d258f61502b2d6'

def fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def parse_episode_seeds(text):
    seeds=[int(part) for part in text.split(',')]
    if not seeds or len(seeds)!=len(set(seeds)):raise ValueError('Episode seeds must be a non-empty unique ordered list')
    return seeds

def select_policy_rows(rows,suite,offset=0,count=None,episode_seeds=None):
    if suite=='seen-training-home':
        if offset!=0 or count not in (None,5):raise ValueError('Seen suite must keep all five layouts')
    elif suite=='seen-training-home-15':
        if offset!=0 or count not in (None,15):raise ValueError('Seen fifteen suite must keep all fifteen layouts')
    elif suite=='heldout-bank-v2':
        if offset not in (0,15) or count!=15:raise ValueError('Held-out scoring requires a fixed fifteen-seed chunk')
    else:raise ValueError('Unknown policy suite')
    selected=rows[offset:offset+count] if count is not None else rows[offset:]
    if len(selected)!=(5 if suite=='seen-training-home' else 15):raise ValueError('Incomplete policy chunk')
    if episode_seeds is not None:
        requested=list(episode_seeds)
        if not requested or len(requested)!=len(set(requested)):raise ValueError('Resume seeds must be non-empty and unique')
        filtered=[row for row in selected if row['seed'] in requested]
        if [row['seed'] for row in filtered]!=requested:raise ValueError('Resume seeds must be an ordered subset of the fixed chunk')
        selected=filtered
    return selected

def load_bank(root,expected_sha256):
    root=Path(root).resolve();path=root/'bank-manifest.json';raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Bank manifest SHA differs')
    bank=json.loads(raw)
    def checked(name,sha):
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Bank file SHA/path differs: '+name)
        return path
    for name,sha in bank['files_sha256'].items():checked(name,sha)
    seen15=bank.get('kind')=='next120-seen-training-home'
    legacy=bank.get('kind')=='seen-training-start-diagnostic'
    if legacy or seen15:
        expected=list(SEEN15_SEEDS) if seen15 else list(range(530000,530005))
        if bank.get('mode')!='home' or bank['seeds']!=expected:raise ValueError('Unqualified seen home bank')
        if not {f'{kind}-{seed}.json' for seed in bank['seeds'] for kind in ('state','seed')}<=bank['files_sha256'].keys():raise ValueError('Missing sealed home file')
        if seen15:
            if not (bank.get('complete') is True and bank.get('format_version')==1 and bank.get('global_episode_indices')==list(range(15)) and bank.get('colour_counts')==dict(red=5,green=5,blue=5)):raise ValueError('Seen fifteen cohort contract differs')
            if fingerprint(Path(bank['source_audit_path']))!=bank['source_audit_sha256']:raise ValueError('Seen fifteen source audit SHA differs')
        rows=[dict(json.loads((root/f'state-{seed}.json').read_text()),bodies=json.loads((root/f'seed-{seed}.json').read_text())) for seed in bank['seeds']]
        for index,row in enumerate(rows):
            receipt=Path(row['source_receipt']);frames=Path(row['source_frames']);reference=receipt.parent.parent/'run.json'
            if fingerprint(receipt)!=row['source_receipt_sha256'] or fingerprint(frames)!=row['source_frames_sha256']:raise ValueError('Training source file SHA differs')
            source=json.loads(receipt.read_text())
            if not (source['accepted'] and source['params']['seed']==row['seed'] and source['params']['color']==row['color'] and source['params']['task']==row['task'] and source['params']['recovery']=='none'):raise ValueError('Training receipt contract differs')
            reference_sha=fingerprint(reference)
            if seen15:
                ref=bank['reference_runs'][row['source_batch']]
                if row['global_episode_index']!=index or row['reference_run']!=str(reference) or row['reference_run_sha256']!=reference_sha or ref!=dict(path=str(reference),sha256=reference_sha):raise ValueError('Seen fifteen per-batch reference differs')
                prepared=source['episode_start_physical']
                for key,source_key in (('state','state'),('recorded_drive_targets','drive_targets'),('robot_base_world_pose','robot_base_world_pose')):
                    if row[key]!=prepared[source_key]:raise ValueError('Seen fifteen recorded home differs')
                for name,pose in row['bodies'].items():
                    if pose!={k:prepared['bodies'][name][k] for k in ('position','orientation')}:raise ValueError('Seen fifteen recorded bodies differ')
            row.update(reference_run=str(reference),reference_run_sha256=reference_sha,first_frame_reference=str(frames))
        suite='seen-training-home-15' if seen15 else 'seen-training-home'
    else:
        if not (bank.get('format_version')==2 and bank.get('purpose')=='held-out-collector-scenes' and bank.get('complete') is True and bank.get('producer')=='collect.Collector.run_scene_bank' and bank['seeds']==list(range(900000,900030))):raise ValueError('Unqualified collector held-out bank')
        for name,sha in bank['sampler_metadata_sha256'].items():checked(name,sha)
        if not {'metadata/provenance.json','metadata/finalization.json'}<=bank['sampler_metadata_sha256'].keys():raise ValueError('Missing bank provenance/finalization')
        provenance=json.loads((root/'metadata/provenance.json').read_text());context=provenance['context'];reference=Path(provenance['reference_run'])
        if fingerprint(reference)!=provenance['reference_run_sha256']:raise ValueError('Sampler reference run SHA differs')
        reference_data=json.loads(reference.read_text())
        if reference_data['baseline']!=provenance['baseline'] or reference_data['baseline_cup_layout']!=provenance['baseline_cup_layout']:raise ValueError('Sampler reference baseline differs')
        if not (context['git_commit']=='bf7336d93d5ba054c5edc66cea685725eb30eb40' and context['mode']=='eval' and provenance['scene_bank_source_sha256']==HELPER_SHA and context['source_sha256']['linux-env/scripts/collect/collect.py']==COLLECTOR_SHA and context['source_sha256']['linux-env/scripts/collect/scene_bank.py']==HELPER_SHA):raise ValueError('Sampler code provenance differs')
        if json.loads((root/'metadata/finalization.json').read_text())['passed'] is not True:raise ValueError('Bank producer did not finalize')
        rows=[]
        for seed in bank['seeds']:
            name=f'metadata/seed-{seed}.json'
            if name not in bank['sampler_metadata_sha256'] or f'seed-{seed}.json' not in bank['files_sha256']:raise ValueError('Missing sealed scene receipt/body file')
            receipt=json.loads((root/name).read_text());params=receipt['params'];physical=receipt['initial_physical'];color=COLORS[seed%3]
            if not (receipt['accepted'] is True and receipt['scene_bank_only'] is True and params==bank['parameters'][str(seed)] and params['seed']==seed and params['color']==color and params['split']=='eval' and params['recovery']=='none' and params['task']==f'Pick a {color} lego and place it in {color} cup'):raise ValueError('Scene receipt contract differs')
            bodies=json.loads((root/f'seed-{seed}.json').read_text())
            for obj,pose in bodies.items():
                if pose!={k:physical['bodies'][obj][k] for k in ('position','orientation')}:raise ValueError('Scene bodies differ from sampler receipt')
            prepared=receipt['episode_start_physical']
            if abs(prepared['sim_ns']-physical['sim_ns']-6e9/30)>1000 or prepared['epoch']!=physical['epoch']:raise ValueError('Collector verification-frame offset differs')
            rows.append(dict(seed=seed,color=color,task=params['task'],state=prepared['state'],recorded_drive_targets=prepared['drive_targets'],robot_base_world_pose=prepared['robot_base_world_pose'],bodies=bodies,source_receipt=str(root/name),source_receipt_sha256=bank['sampler_metadata_sha256'][name],reference_run=str(reference),reference_run_sha256=provenance['reference_run_sha256'],first_frame_reference=None))
        suite='heldout-bank-v2'
    for seed,row in zip(bank['seeds'],rows):
        if row['seed']!=seed or len(row['state'])!=6 or len(row['recorded_drive_targets'])!=6 or not all(math.isfinite(v) for v in row['state']+row['recorded_drive_targets']):raise ValueError('Scene home state contract differs')
        if set(row['bodies'])!=NAMES:raise ValueError('Scene object set differs')
        row.update(bank_root=str(root),bank_sha256=expected_sha256,suite=suite)
    return rows,suite

def reference_binding(rows,suite):
    mapping={str(row['seed']):row['reference_run_sha256'] for row in rows}
    if suite=='seen-training-home-15':
        if [row['seed'] for row in rows]!=list(SEEN15_SEEDS):raise ValueError('Reference binding requires the complete seen fifteen cohort')
        return dict(reference_sha256_by_seed=mapping)
    refs=set(mapping.values())
    if len(refs)!=1:raise ValueError('This suite requires a single reference run')
    return dict(reference_sha256=next(iter(refs)))
