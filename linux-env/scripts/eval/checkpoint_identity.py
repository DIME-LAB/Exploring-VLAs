"""Names distinguish primary training steps from additional unfreeze steps."""
from pathlib import Path
def checkpoint_label(model):
    model=Path(model);step=int(model.parent.name)
    return f'unfreeze30k+{step}' if model.parent.parent.name=='unfreeze30k' else f'{step//1000}k'

def validate_handoff_identity(handoff,model,step,source_checkpoints):
    """Bind verified bytes to one source job, step and published destination."""
    model=Path(model);source=str(Path(source_checkpoints)/f'{step:06d}'/'pretrained_model')
    if handoff['step']!=step:raise ValueError('Handoff step differs from preset')
    if handoff['published_model']!=str(model):raise ValueError('Handoff destination differs from preset')
    if handoff['source']!=source:raise ValueError('Handoff source job/step differs')
    for key in ('source_before','source_after'):
        probe=handoff[key]
        if not probe['ready'] or probe['step']!=step or probe['model']!=source:raise ValueError('Handoff source probe identity differs')
    if handoff['source_before']['files']!=handoff['source_after']['files']:raise ValueError('Handoff source changed during transfer')
    if handoff['verified']['verified'] is not True:raise ValueError('Handoff target was not verified')
    for name,row in handoff['source_before']['files'].items():
        target=handoff['verified']['files'][name]
        if any(row[k]!=target[k] for k in ('size','sha256')):raise ValueError('Handoff target bytes differ: '+name)
