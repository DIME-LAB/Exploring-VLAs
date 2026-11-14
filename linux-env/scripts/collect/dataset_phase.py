"""Per-frame training selection for normal motion and corrective recovery episodes."""
from comparisons import native_frame

PHASES=dict(normal=0,miss_prefix=1,corrective=2)
FEATURE='collection_phase'
POLICY=dict(version=1,feature=FEATURE,encoding=PHASES,train_default=['normal','corrective'],
    mask_default=['miss_prefix'],miss_action='clean expert correction toward the true target; executed miss is separate')


def observed_phase(recovery,epoch,completed_epoch):
    if recovery=='none':return 'normal'
    if recovery!='regrasp':raise ValueError('Unknown recovery kind')
    return 'corrective' if epoch==completed_epoch else 'miss_prefix'


def phase_code(row,receipt,required=False):
    if receipt['params'].get('recovery','none')=='none':expected='normal'
    elif receipt['recovery_report'].get('strategy')=='synthetic_post_miss_reset':
        if native_frame(row)['sim_ns']<=receipt['recovery_report']['record_after_sim_ns']:raise ValueError('Synthetic setup frame cannot be recorded')
        expected='corrective'
    else:
        end=receipt['recovery_report']['clearance_end_sim_ns']
        expected='miss_prefix' if native_frame(row)['sim_ns']<=end else 'corrective'
    phase=row.get(FEATURE)
    if phase is None and not required and expected=='normal':phase='normal'
    if phase!=expected:raise ValueError('Frame phase differs from recovery boundary: '+str((phase,expected)))
    return PHASES[phase]


def feature():return dict(dtype='int64',shape=(1,),names=['phase'])
