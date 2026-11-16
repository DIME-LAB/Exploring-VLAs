"""Native observation evidence for triplet preparation and recording boundaries."""
import copy,math
from contract import COLORS,visible_target
from scene_identity import identity,require_identical,require_pose_agreement,compare
from comparisons import matches
from tick_contract import TickContract,STEP_NS,CLOCK_POLICY,digest,require_snapshots


def clocks(snapshot):
    epoch,stamp,world=(snapshot[k] for k in ('epoch','sim_ns','physics_time_s'))
    if (type(epoch) is not int or epoch<=0 or type(stamp) is not int or stamp<=0 or
            not isinstance(world,(int,float)) or not math.isfinite(world) or world<=0):raise ValueError('Invalid native triplet clock identity')
    return epoch,stamp,world


def from_row(row):
    return dict(epoch=row['epoch'],sim_ns=row['pose_stamp_ns'],physics_time_s=row['physics_time_s'],state_source=row['state_source'],
        state=row['physics_state'],velocity=row['physics_velocity'],drive_targets=row['physics_drive_targets'],bodies=row['physical_bodies'],
        base_world_pose=row['base_world_pose'],robot_base_world_pose=row['robot_base_world_pose'])


def checked(snapshot):
    clocks(snapshot);require_snapshots([snapshot])
    row=snapshot['tick_contract']['row'];require_identical(from_row(row),snapshot)
    return row


def selections(snapshot,anchor):
    evidence={color:visible_target(snapshot,color,True) for color in COLORS}
    for color,proof in evidence.items():
        if proof['object']!=visible_target(anchor,color,True)['object']:raise ValueError('Observed visible target differs from anchor: '+color)
    return dict(physical_sha256=identity(snapshot),native_clock=list(clocks(snapshot)),selections=evidence)


def bootstrap(restored,check,observed):
    clocks(restored);checked(observed)
    audit=TickContract();audit.install_boundary(dict(check,snapshot=restored),restore=True)
    row=copy.deepcopy(observed['tick_contract']['row']);audit.observe(row)
    if audit.finish(row)!=observed['tick_contract']:raise ValueError('Bootstrap proof differs from its restore boundary')
    if restored['epoch']!=observed['epoch'] or not matches(observed['sim_ns']-restored['sim_ns'],STEP_NS,'native_ns'):
        raise ValueError('Bootstrap did not advance one native frame from restore')


def restore_admission(restored,check,anchor):
    proof=check['triplet_physical_state'];reread=proof['reread_snapshot']
    if clocks(restored)!=clocks(reread):raise ValueError('Held restore reread changed native clocks')
    expected=dict(against_target=require_pose_agreement(restored,anchor),reread_snapshot=reread,
        reread_exact_bits=compare(reread,restored),reread_agreement=require_pose_agreement(reread,restored),
        reread_against_target=require_pose_agreement(reread,anchor),native_clocks_unchanged=True)
    if proof!=expected:raise ValueError('Restore pose/reread evidence differs')
    return expected


def restore_prior(current,previous):
    def reads(r):return [r['snapshot'],r['triplet_physical_state']['reread_snapshot']]
    return [dict(previous_member=i,current_read=j,previous_read=k,proof=require_pose_agreement(a,b))
        for i,old in enumerate(previous) for j,a in enumerate(reads(current)) for k,b in enumerate(reads(old))]


def later(current,previous,prime_ticks=20):
    epoch,stamp,world=clocks(current);old_epoch,old_stamp,old_world=clocks(previous)
    native_delta=stamp-old_stamp;world_delta=world-old_world
    native_min=prime_ticks*STEP_NS;world_min=prime_ticks/30
    world_tolerance=prime_ticks*max(CLOCK_POLICY['world_delta_min_tolerance_s'],
        CLOCK_POLICY['world_delta_ulp_factor']*math.ulp(max(abs(world),abs(old_world),1.)))
    if (epoch<=old_epoch or world_delta<=0 or
            native_delta<native_min and not matches(native_delta,native_min,'native_ns') or
            world_delta<world_min and not matches(world_delta,world_min,'world_s',tolerance=world_tolerance)):
        raise ValueError('Triplet member reused or overlapped its predecessor native boundary')


def recorded(snapshot,row,initial,anchor,episode_id,triplet_id):
    checked(snapshot);checked(initial)
    if row['episode_id']!=episode_id or row['triplet_id']!=triplet_id:raise ValueError('Native start belongs to another episode/group')
    raw={k:v for k,v in row.items() if k!='wall_profile'}
    if raw!=snapshot['tick_contract']['row']:raise ValueError('Recorded start differs from its native tick proof')
    require_identical(from_row(row),snapshot)
    if snapshot['epoch']!=initial['epoch']:raise ValueError('Recorded start crosses its bootstrap epoch')
    offset=snapshot['sim_ns']-initial['sim_ns']
    if offset==0:
        require_identical(snapshot,initial)
        if snapshot['tick_contract']!=initial['tick_contract']:raise ValueError('Same-tick start contradicts bootstrap proof')
    elif matches(offset,STEP_NS,'native_ns'):
        require_snapshots([initial,snapshot])
        if snapshot['tick_contract']['previous']!=initial['tick_contract']['row']:raise ValueError('Next-tick start omitted bootstrap predecessor')
    else:raise ValueError('Recorded start is neither bootstrap nor its next native tick')
    return dict(version=1,episode_id=episode_id,triplet_id=triplet_id,bootstrap_sha256=digest(initial),
        native=copy.deepcopy(snapshot),observed_rule=selections(snapshot,anchor))


def verify_record(record,row,initial,anchor,episode_id,triplet_id):
    expected=recorded(record['native'],row,initial,anchor,episode_id,triplet_id)
    if record!=expected:raise ValueError('Native start certificate differs')
    return expected
