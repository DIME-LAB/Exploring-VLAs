"""Validate an observed native invocation and the completed synthetic setup prefix."""
import ast,copy,re
from pathlib import Path
from comparisons import matches,interval
from tick_contract import digest,require_snapshots,STEP_NS,world_step_errors
from recovery_start import (POLICY,STRATEGY,StaticClearance,candidates,geometry,world_anchor,
    replay_baseline,replay_model_binding,restore_world_checks,protected_state)
from miss_clearance import bind_target_offsets,proof_differences
from scene_sync import physical_pose_caches
from contract import JOINTS
from so_arm101_control.grasp_geometry import JOINT_LIMITS


def semantic_error(value):
    """Unwrap exception/string renderings without executing receipt text."""
    for _ in range(8):
        try:node=ast.parse(value,mode='eval').body
        except (SyntaxError,TypeError):return value
        if isinstance(node,ast.Constant) and isinstance(node.value,str):value=node.value;continue
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and
                node.func.id in ('RuntimeError','OwnershipStalled') and len(node.args)==1 and not node.keywords and
                isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str)):
            value=node.args[0].value;continue
        return value
    return value


def verify_transport(observation,boot,stage,runtime,receipt,fault=None):
    if observation.get('version')!=1 or not re.fullmatch('[0-9a-f]{32}',str(observation.get('invocation'))):
        raise RuntimeError('Independent native invocation observation is missing')
    if any(observation['stage'].get(k)!=stage.get(k) for k in ('pid','started_wall_ns','invocation')):
        raise RuntimeError('Native observation belongs to another producing stage')
    if observation['invocation']!=stage.get('invocation'):raise RuntimeError('Native invocation nonce differs')
    before,after=observation['before'],observation['after'];native=boot['native'];transport=boot['transport']
    expected_keys=('pid','start_ticks','pgid','fault_path','loaded_sha256','runtime_module')
    if any(before.get(k)!=native.get(k) or after.get(k)!=native.get(k) for k in expected_keys):
        raise RuntimeError('Native startup/transport identity changed')
    if any(v.get('owned_identity')!={k:native[k] for k in ('pid','start_ticks','pgid')} for v in (native,before,after)):
        raise RuntimeError('Native transport is not a recorded spawned process')
    if before.get('nonce')!=observation['invocation'] or after.get('nonce')!=observation['invocation']:
        raise RuntimeError('Native observation round trips are unbound')
    if not before['wall_ns']<=stage['started_wall_ns']<=after['wall_ns']:
        raise RuntimeError('Native observation does not bracket the producer')
    if before.get('fault') is not None or before.get('failure') or before.get('busy') or before.get('playing'):
        raise RuntimeError('Native startup was already unhealthy or moving')
    if after.get('busy') or after.get('playing'):raise RuntimeError('Native transport did not retire held')
    if (transport['native_miss_fault_path']!=native['fault_path'] or
            transport['loaded_sha256']!=native['loaded_sha256'] or
            not Path(native['fault_path']).is_relative_to(runtime/'native-motion-faults')):
        raise RuntimeError('Native boot configuration differs from its captured transport')
    report=receipt['recovery_report'];baseline=report['baseline']
    if not before['epoch']<baseline['epoch']<=after['epoch'] or not interval(baseline['sim_ns'],before['sim_ns'],after['sim_ns'],'native_ns'):
        raise RuntimeError('Synthetic baseline is outside its native invocation')
    for sample in (before,after):
        snap=sample['snapshot']
        if snap['epoch']!=sample['epoch'] or not matches(snap['sim_ns'],sample['sim_ns'],'native_ns'):
            raise RuntimeError('Independent native readback clock/epoch differs')
    if after['snapshot']['local_corners']!=baseline['local_corners']:
        raise RuntimeError('Synthetic protected geometry differs from native readback')
    if fault is None:
        if after.get('fault') is not None or after.get('failure'):raise RuntimeError('A late native fault invalidates success')
    else:
        if semantic_error(after.get('failure'))!=semantic_error(fault['error']):
            raise RuntimeError('Another native terminal cause is present')
        if (receipt['native_miss_fault_file']!=native['fault_path'] or
                digest(after.get('fault'))!=digest(fault) or after.get('fault_durable') is not True):
            raise RuntimeError('Native journal is not the current independently observed fault')
        if not before['epoch']<fault['epoch']<=after['epoch']:
            raise RuntimeError('Native fault epoch is outside the producing invocation')
        if not interval(fault['sim_ns'],before['sim_ns'],after['sim_ns'],'native_ns'):
            raise RuntimeError('Native fault clock is outside the producing invocation')
        physical=protected_state(report['baseline'],after['snapshot'])
        if (not all(c['passed'] for c in physical['base_checks']) or
                any(v['max_corner_motion_m']>POLICY['protected_corner_motion_max_m'] for v in physical['bodies'].values())):
            raise RuntimeError('Current native physical scene differs from the failing setup')
    return transport


def setup_prefix(receipt,transport,stage):
    """Replay the complete validator's checks up to the one failing screen."""
    report=receipt['recovery_report'];baseline=report['baseline'];observed=report['pre_restore_observed']
    if report['strategy']!=STRATEGY or report['policy']!=POLICY or report['origin']!=POLICY['origin']:
        raise RuntimeError('Unknown synthetic setup identity')
    require_snapshots([observed])
    expected=replay_baseline(observed,report['replay_world_anchor']) if 'replay_world_anchor' in report else observed
    if baseline!=expected or report['world_anchor_sha256']!=digest(world_anchor(baseline)):
        raise RuntimeError('Synthetic pre-restore world anchor differs')
    fresh=report['fresh_offsets'];target=receipt['params']['object']
    if (fresh['epoch']!=baseline['epoch'] or not matches(fresh['sim_ns'],baseline['sim_ns'],'native_ns') or
            fresh['finger_offsets']!=transport['finger_model']['effective_finger_offsets'] or
            fresh['block_offsets']!=transport['effective_shape_offsets'][target]):
        raise RuntimeError('Fresh model/offset readback differs from the native transport')
    model=bind_target_offsets(transport['finger_model'],fresh['block_offsets'])
    if not replay_model_binding(report['model'],model)['passed']:raise RuntimeError('Synthetic robot model differs from native startup')
    if 'replay_world_anchor' in report:
        binding=replay_model_binding(report['model'],report['replay_robot_model'])
        if not binding['passed'] or report['replay_model_binding']!=binding:raise RuntimeError('Replay model binding differs')
    poses,_=physical_pose_caches(baseline,primitive_cups=True)
    if proof_differences(report['request']['pose'],poses[target],'synthetic.request.pose'):
        raise RuntimeError('Synthetic request pose is not the measured target')
    corners=baseline['local_corners'][target]
    dims=[max(c[i] for c in corners)-min(c[i] for c in corners) for i in range(3)]
    if not matches([report['request']['bbox'][n] for n in ('sx','sy','sz')],dims,'float32_readback'):
        raise RuntimeError('Synthetic request dimensions differ')
    oracle=StaticClearance(model,baseline,target)
    if stage=='baseline':
        if report['epoch']!=baseline['epoch']:raise RuntimeError('Precondition screen epoch differs')
        return oracle
    selection=report['selection'];attempts=selection['attempts'];specs=list(candidates(receipt['params']['regrasp']));index=selection['spec']['index']
    if (len(attempts)!=index+1 or selection['spec']!=specs[index] or
            any(a['spec']!=specs[i] for i,a in enumerate(attempts)) or
            any(a['passed'] for a in attempts[:-1]) or not attempts[-1]['passed'] or not selection['proof']['passed']):
        raise RuntimeError('Synthetic first-passing selection proof differs')
    g=geometry(report['request']['pose'],report['request']['bbox'],selection['spec'],report['request']['settings'])
    if proof_differences(selection['geometry'],g,'synthetic.geometry') or not any(matches(selection['joints'],q,'arithmetic') for q in g['solutions']):
        raise RuntimeError('Synthetic IK selection differs')
    if (not matches(selection['proof']['joints'],selection['joints'],'arithmetic') or
            not any(v==selection['proof'] for v in attempts[-1]['verdicts']) or
            any(not JOINT_LIMITS[n][0]<=x<=JOINT_LIMITS[n][1] for n,x in zip(JOINTS,selection['joints']))):
        raise RuntimeError('Synthetic selected command differs from its verdict/limits')
    static=oracle.state(selection['joints'])
    if not static['passed'] or proof_differences(selection['proof']['static'],static,'synthetic.static'):
        raise RuntimeError('Synthetic selected pose is not statically certified')
    restore=report['restore'];snapshot=restore['snapshot'];command=restore['command_baseline'];epoch=report['epoch']
    if (restore['epoch']!=epoch or snapshot['epoch']!=epoch or command['epoch']!=epoch or epoch<=baseline['epoch'] or
            not matches(snapshot['sim_ns'],baseline['sim_ns'],'native_ns') or
            not matches(restore['invalid_through_ns'],snapshot['sim_ns']) or not matches(command['sim_ns'],snapshot['sim_ns']) or
            not matches(snapshot['physics_time_s'],baseline['physics_time_s'],'world_s') or
            not matches(command['physics_time_s'],snapshot['physics_time_s'])):
        raise RuntimeError('Synthetic restore epoch/time/command boundary differs')
    if (not restore['arm_restored'] or not matches(snapshot['state'],selection['joints'],'target_rad') or
            not matches(snapshot['drive_targets'],selection['joints'],'target_rad') or
            not matches(snapshot['velocity'],[0.]*6,'float32_readback') or
            not matches(command['drive_targets'],snapshot['drive_targets']) or
            any(not c['passed'] for c in restore_world_checks(baseline,snapshot))):
        raise RuntimeError('Synthetic native restore/held targets differ')
    if report['protected_start']!=dict(epoch=epoch,sim_ns=snapshot['sim_ns'],anchor_sha256=report['world_anchor_sha256']):
        raise RuntimeError('Protected start does not bind the restore')
    ticks=report.get('settle_snapshots',[]);checks=report.get('settle_checks',[])
    if len(ticks)!=len(checks) or len(ticks)>POLICY['settle_ticks']:raise RuntimeError('Returned setup prefix has missing checks')
    if ticks:
        require_snapshots(ticks);first=ticks[0]['tick_contract']['row'];drive=first['drive_baseline']
        if (drive.get('kind')!='restore' or any(drive.get(k)!=v for k,v in command.items()) or
                not matches(first['physics_drive_targets'],command['drive_targets'],'target_rad') or
                not matches(ticks[0]['sim_ns']-snapshot['sim_ns'],STEP_NS,'native_ns') or
                world_step_errors(ticks[0]['physics_time_s'],snapshot['physics_time_s'])):
            raise RuntimeError('First settle tick is not the verified restore boundary')
    for tick,proof in zip(ticks,checks):
        if tick['epoch']!=epoch or not matches(tick['drive_targets'],tick['tick_contract']['row']['physics_drive_targets']):
            raise RuntimeError('Returned setup drive readback differs')
        expected=dict(static=oracle.state(tick['state'],tick),target=oracle.target_state(tick),protected=protected_state(baseline,tick))
        if not proof['scene_valid'] or any(not v['passed'] or proof_differences(proof[k],v,'synthetic.settle.'+k) for k,v in expected.items()):
            raise RuntimeError('Returned setup static/scene evidence differs')
    if 'scene_after_reset' in report and not report['scene_after_reset']['passed']:
        raise RuntimeError('Synthetic planning scene differs')
    return oracle
