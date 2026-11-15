"""Read-only full-checkpoint fingerprints for native evaluation."""
import hashlib,json
from pathlib import Path

def checkpoint_files(directory):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('Checkpoint directory must not be a symlink')
    result={}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Checkpoint contains a symlink')
        if not path.is_file():continue
        before=path.stat();digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Checkpoint changed while hashing')
        result[str(path.relative_to(directory))]=dict(size=before.st_size,sha256=digest.hexdigest())
    return result

def require_handoff(directory,receipt_path,weights_sha256,*,published=False):
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt['source_before']['files']!=receipt['source_after']['files'] or receipt['verified']['verified'] is not True:raise ValueError('Handoff lacks stable verified source files')
    if published:
        if not (receipt['source_before']['ready'] and receipt['source_after']['ready'] and receipt['stable_seconds']>=60):raise ValueError('Published handoff lacks source completion/stability')
        expected_path=receipt['published_model']
    else:expected_path=receipt['source_before']['model']
    if Path(directory).resolve()!=Path(expected_path).resolve():raise ValueError('Evaluation source path differs from handoff')
    actual=checkpoint_files(directory)
    expected={name:{key:row[key] for key in ('size','sha256')} for name,row in receipt['source_before']['files'].items()}
    if published and {name:{key:row[key] for key in ('size','sha256')} for name,row in receipt['verified']['files'].items()}!=expected:raise ValueError('Published verification differs from stable source')
    if actual!=expected:raise ValueError('Checkpoint files differ from verified handoff')
    if actual['model.safetensors']['sha256']!=weights_sha256:raise ValueError('Checkpoint weights differ')
    return dict(receipt=str(receipt_path),receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),files=actual)

def require_processor_proof(directory,proof_path,files,*,published_handoff=None):
    directory=Path(directory);proof_path=Path(proof_path);proof=json.loads(proof_path.read_text())
    if 'actual_training_checkpoint' in proof:
        if not (proof['passed'] is True and proof['cpu_only'] is True and proof['actual_training_checkpoint'] is True and proof['fixture_only'] is False):raise ValueError('Actual saved training processor proof required')
        if Path(proof['checkpoint']).resolve()!=directory.resolve():raise ValueError('Processor proof belongs to another checkpoint')
        if published_handoff is None:raise ValueError('Robustness processor proof requires completed published handoff')
        handoff=json.loads(Path(published_handoff).read_text())
        expected={n:{k:v[k] for k in ('size','sha256')} for n,v in handoff['source_before']['files'].items()}
        verified={n:{k:v[k] for k in ('size','sha256')} for n,v in handoff['verified']['files'].items()}
        if not (handoff['source_before']['ready'] and handoff['source_after']['ready'] and handoff['verified']['verified'] and handoff['stable_seconds']>=60 and handoff['source_before']['files']==handoff['source_after']['files'] and expected==verified==files):raise ValueError('Processor proof handoff is incomplete or differs')
        if Path(handoff['source_before']['model']).resolve()!=directory.resolve():raise ValueError('Processor source differs from completed handoff')
        completion=handoff['source_before']['completion']
        if completion['training_step_present'] is not True or int(completion['target'])<int(directory.parent.name):raise ValueError('Trainer completion marker has not reached this checkpoint')
        if proof['crop_module_sha256']!=files['workspace_crop_step.py']['sha256']:raise ValueError('Processor module differs from proof')
        required={'observation.images.top','observation.images.wrist','observation.language.tokens','observation.state','top_after_resize'}
        if set(proof['pixel_state_token_resize_hashes'])!=required or any(len(v)!=64 for v in proof['pixel_state_token_resize_hashes'].values()):raise ValueError('Incomplete processor parity hashes')
        if proof['raw_image_shapes']!=[3,480,640] or proof['inference_order']!=['top_crop','normal_resize'] or proof['no_random_inference_augmentation'] is not True:raise ValueError('Inference processor image contract differs')
        copied=Path(handoff['published_model'])/'policy_preprocessor.json'
        if hashlib.sha256(copied.read_bytes()).hexdigest()!=files['policy_preprocessor.json']['sha256']:raise ValueError('Saved processor differs from full handoff')
        steps=json.loads(copied.read_text())['steps'];classes=[s['class'] for s in steps if 'class' in s]
        if classes!=['workspace_crop_step.WorkspaceTopCropStep']:raise ValueError('Unexpected inference augmentation/custom processor')
        if [s.get('class',s.get('registry_name')) for s in steps]!=['rename_observations_processor','to_batch_processor','workspace_crop_step.WorkspaceTopCropStep','smolvla_new_line_processor','tokenizer_processor','device_processor','normalizer_processor']:raise ValueError('Inference crop/processor order differs')
        registries=[s['registry_name'] for s in steps if 'registry_name' in s]
        if registries!=['rename_observations_processor','to_batch_processor','smolvla_new_line_processor','tokenizer_processor','device_processor','normalizer_processor']:raise ValueError('Unexpected inference registry step')
        crop=next(s['config'] for s in steps if 'class' in s)
        if crop!={'key':'observation.images.top','box_xyxy':[99,128,559,473],'input_hw':[480,640]} or steps[-1]['config']['norm_map']['VISUAL']!='IDENTITY':raise ValueError('Saved deterministic crop/visual normalisation differs')
        return dict(path=str(proof_path),sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),schema='robustness-saved-crop',proof=proof)
    if not (proof['passed'] is True and proof['cpu_only'] is True and proof['training_written_processor_reloaded'] is True and proof['fixture_only'] is False):raise ValueError('Actual saved training processor proof required')
    if Path(proof['checkpoint']).resolve()!=directory.resolve() or Path(proof['completed_last_marker']).resolve()!=directory.parent.resolve():raise ValueError('Processor proof belongs to another checkpoint')
    if proof['module_sha256']!=files['workspace_crop_step.py']['sha256']:raise ValueError('Processor module differs from proof')
    if proof['raw_policy_input_keys']!=['observation.state','observation.images.top','observation.images.wrist']:raise ValueError('Processor input keys differ')
    for name,size in proof['stable_file_sizes'].items():
        if files[name]['size']!=size:raise ValueError('Processor proof file size differs')
    required={'observation.images.top','observation.images.wrist','observation.language.tokens','observation.state','top_after_native_resize'}
    hashes=proof['pixel_state_token_and_resize_hashes']
    if set(hashes)!=required or any(len(value)!=64 for value in hashes.values()):raise ValueError('Incomplete processor parity hashes')
    return dict(path=str(proof_path),sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),proof=proof)
