"""Pipeline checks for every physical tick, including unrecorded verdict ticks."""
import copy,hashlib,json,math,struct
from comparisons import comparison,matches,interval,bound_comparison,TOLERANCES

STEP_NS=round(1e9/30)
PHYSICS_DT=1/120
PHYSICS_STEPS=4
FLOAT32_DT=struct.unpack('f',struct.pack('f',PHYSICS_DT))[0]
CLOCK_POLICY=dict(version=3,agreement_rules=TOLERANCES,cadence_clock='pose_stamp_ns',native_step_ns=STEP_NS,physics_dt=PHYSICS_DT,
    physics_steps_per_frame=PHYSICS_STEPS,world_delta_min_tolerance_s=1e-6,
    world_delta_ulp_factor=8,absolute_world_native_equality=False)


def world_step_errors(current,previous):
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (current,previous)):
        return ['World clock is missing or non-finite']
    # World accumulates the PhysX callback step_size; graph/camera stamps use
    # Isaac Read Simulation Time. Four float32 steps plus double accumulation
    # need not have the same absolute origin/value as the native graph clock.
    tolerance=max(1e-6,8*math.ulp(max(abs(current),abs(previous),1.))+
                  PHYSICS_STEPS*abs(FLOAT32_DT-PHYSICS_DT))
    if tolerance>=PHYSICS_DT/4:return ['World clock precision cannot resolve a physics substep']
    delta=current-previous
    if delta<=0:return ['World clock is frozen or rewound']
    if not matches(delta,PHYSICS_STEPS*PHYSICS_DT,'world_s',tolerance=tolerance):return ['World clock did not advance one frame']
    return []


class TickContractError(RuntimeError):
    def __init__(self,errors,evidence=None):
        self.errors=list(errors);self.evidence=evidence
        super().__init__('Physical tick contract failed: '+'; '.join(self.errors))


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def vector(value,size):
    return isinstance(value,(list,tuple)) and len(value)==size and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value)


def camera_tick_errors(row,required=False):
    identities=row.get('camera_identity',{})
    if not isinstance(identities,dict):return ['Invalid native camera selection identities']
    if not required and not any(isinstance(identity,dict) and 'native_tick' in identity for identity in identities.values()):return []
    errors=[]
    for key in ('wrist','top'):
        identity=identities.get(key,{})
        native=identity.get('native_tick') if isinstance(identity,dict) else None
        if not isinstance(native,dict):
            errors.append('Missing native camera selection proof '+key)
            continue
        for proof in (comparison(key+' selection epoch',native.get('epoch'),row.get('epoch')),
                comparison(key+' native selection tick',native.get('sim_ns'),row.get('pose_stamp_ns')),
                comparison(key+' selected raw header',identity.get('stamp_ns'),row.get(key+'_stamp_ns')),
                comparison(key+' camera/native producer agreement',identity.get('stamp_ns'),native.get('sim_ns'),'stamp_ns')):
            if not proof['passed']:errors.append('Camera selection contract: '+json.dumps(proof,sort_keys=True))
    return errors


def observation_errors(row,previous=None):
    errors=camera_tick_errors(row)
    if previous is not None:
        errors.extend('Previous tick: '+error for error in observation_errors(previous))
    stamp=row.get('sim_ns',0);epoch=row.get('epoch')
    if not isinstance(stamp,int) or stamp<=0:return ['Missing positive native sim stamp']
    if epoch is None or any(row.get(k)!=epoch for k in ('observation_epoch','expert_epoch')):errors.append('Tick epoch mismatch')
    if stamp<=row.get('invalid_through_ns',stamp):errors.append('Tick is at/before reset boundary')
    for key in ('joint_stamp_ns','wrist_stamp_ns','top_stamp_ns','pose_stamp_ns'):
        if not matches(row.get(key),stamp,'stamp_ns'):errors.append(key+' differs from sim tick')
    if row.get('state_source')!='isaac_articulation':errors.append('Tick state is not measured')
    for key in ('state','velocity','physics_state','physics_velocity','physics_drive_targets'):
        if not vector(row.get(key),6):errors.append('Invalid '+key)
    if not errors:
        for a,b in (('state','physics_state'),('velocity','physics_velocity')):
            if not matches(row[a],row[b],'state_rad'):errors.append(a+' differs from physical snapshot')
    for camera in ('wrist','top'):
        layout=row.get('camera_layout',{}).get(camera,{})
        if (layout.get('encoding'),layout.get('width'),layout.get('height'))!=('rgb8',640,480) or layout.get('step',0)<1920 or layout.get('bytes')!=480*layout.get('step',0):errors.append('Invalid camera layout '+camera)
    physical=row.get('physics_time_s')
    if not isinstance(physical,(int,float)) or not math.isfinite(physical):
        errors.append('World clock is missing or non-finite')
    for name,body in row.get('physical_bodies',{}).items():
        if not all(vector(body.get(k),size) for k,size in (('position',3),('orientation',4),('velocity',6))):
            errors.append('Invalid physical body '+name)
        elif sum(v*v for v in body['orientation'])<1e-10:errors.append('Invalid body quaternion '+name)
    for key in ('wall_ns','joint_receive_ns','wrist_receive_ns','top_receive_ns'):
        if not isinstance(row.get(key),int) or row[key]<=0:errors.append('Missing wall receipt '+key)
    if previous and previous.get('epoch')==epoch:
        if not matches(row['pose_stamp_ns']-previous['pose_stamp_ns'],STEP_NS,'native_ns'):errors.append('Repeated or skipped physical tick')
        errors.extend(world_step_errors(physical,previous.get('physics_time_s')))
        for key in ('wall_ns','wrist_receive_ns','top_receive_ns'):
            if row.get(key,0)<=previous.get(key,0):errors.append('Non-monotonic '+key)
        if vector(row.get('physics_drive_targets'),6) and vector(previous.get('applied_command'),6):
            if not matches(row['physics_drive_targets'],previous['applied_command'],'target_rad'):
                errors.append('Previously ACKed targets did not persist')
    elif row.get('drive_baseline') is not None:
        baseline=row['drive_baseline']
        if baseline.get('epoch')!=epoch or not matches(baseline.get('sim_ns'),row.get('invalid_through_ns')):
            errors.append('Drive baseline belongs to another restore/pause boundary')
        if not vector(baseline.get('drive_targets'),6):errors.append('Drive baseline is incomplete')
        elif vector(row.get('physics_drive_targets'),6) and not matches(row['physics_drive_targets'],baseline['drive_targets'],'target_rad'):
            errors.append('Physical targets differ from the boundary baseline')
        if not matches(row.get('pose_stamp_ns',stamp)-baseline.get('sim_ns',stamp),STEP_NS,'native_ns'):errors.append('First frame skipped the drive baseline')
        if baseline.get('physics_time_s') is not None:errors.extend(world_step_errors(physical,baseline['physics_time_s']))
    return errors


def row_errors(row,previous=None):
    errors=observation_errors(row,previous)
    stamp=row.get('sim_ns',0)
    for key in ('action','executed_action','expert_reference','applied_command'):
        if not vector(row.get(key),6):errors.append('Invalid '+key)
    if not matches(row.get('expert_observation_stamp_ns'),stamp):errors.append('Expert uses another observation stamp')
    if not matches(row.get('observation_stamp_ns'),stamp):errors.append('ACK uses another observation stamp')
    if not matches(row.get('declared_application_stamp_ns'),stamp):errors.append('Command scheduled at another tick')
    latency=row.get('application_stamp_ns',-100)-stamp
    if not interval(latency,0,STEP_NS,'stamp_ns'):errors.append('Application exceeds one-tick contract')
    if not interval(row.get('command_clock_ns',-10000)-stamp,0,STEP_NS,'stamp_ns'):errors.append('Command clock differs from held tick')
    if not errors:
        if not matches(row['executed_action'],row['applied_command'],'target_rad'):errors.append('ACK target mismatch')
        clean=[q+max(-.035,min(.035,r-q)) for q,r in zip(row['state'],row['expert_reference'])]
        if not matches(clean,row['action'],'arithmetic'):errors.append('Noisy or stale expert label')
    if row.get('tick_contract_version'):
        for key,size in (('planner_execution_reference',6),('dart_offset',6),('command_filter_delta',6),('command_velocity',5)):
            if not vector(row.get(key),size):errors.append('Invalid execution provenance '+key)
        if not errors:
            if not matches(row['executed_action'],[r+n for r,n in zip(row['planner_execution_reference'],row['dart_offset'])],'arithmetic'):errors.append('Executed target differs from planner plus offset')
            if any(row['command_filter_delta']) or row['dart_offset'][5]!=0.:errors.append('Expert path or gripper was perturbed')
            velocity=[(q-p)*30 for q,p in zip(row['executed_action'][:5],row['physics_drive_targets'][:5])]
            if not matches(velocity,row['command_velocity'],'velocity_rad_s'):errors.append('Command velocity provenance differs')
    geometry=row.get('execution_geometry',{})
    if not geometry.get('passed') or geometry.get('epoch')!=row.get('epoch') or not matches(geometry.get('sim_ns'),stamp) or not matches(geometry.get('target'),row.get('executed_action')):
        errors.append('Final command geometry identity differs')
    return errors


class TickContract:
    def __init__(self):self.previous=None;self.boundary=None;self.anchor=None
    def install_boundary(self,result,restore=False,pause_targets=None):
        epoch=result['epoch'];stamp=result['invalid_through_ns'];previous=self.previous
        anchor=previous or self.anchor or self.boundary
        anchor_stamp=anchor.get('pose_stamp_ns',anchor['sim_ns']) if anchor else None
        if anchor and (epoch<=anchor['epoch'] or stamp<anchor_stamp-10):
            raise TickContractError(['Cancellation/restore rewound the native clock or replayed an epoch'])
        if restore:
            baseline=copy.deepcopy(result['command_baseline']);snapshot=result['snapshot']
            if (baseline.get('epoch')!=epoch or snapshot.get('epoch')!=epoch or not matches(baseline.get('sim_ns'),stamp) or
                not matches(snapshot.get('sim_ns'),stamp) or not matches(baseline.get('drive_targets'),snapshot.get('drive_targets')) or
                not matches(baseline.get('physics_time_s'),snapshot.get('physics_time_s')) or not vector(baseline.get('drive_targets'),6) or
                not isinstance(result.get('arm_restored'),bool)):
                raise TickContractError(['Restore did not supply its native epoch-stamped target baseline'])
            world=baseline.get('physics_time_s')
            if not isinstance(world,(int,float)) or not math.isfinite(world):raise TickContractError(['Restore World clock is invalid'])
            if previous and world<previous['physics_time_s']:raise TickContractError(['Restore rewound the World clock'])
            baseline['kind']='restore'
            world=baseline['physics_time_s']
        else:
            targets=previous['applied_command'] if previous else pause_targets
            world=(previous['physics_time_s'] if previous and matches(stamp,previous.get('pose_stamp_ns',previous['sim_ns']),'native_ns') else None)
            clock=result.get('clock_baseline')
            if clock is not None:
                world=clock.get('physics_time_s')
                if clock.get('epoch')!=epoch or not matches(clock.get('sim_ns'),stamp) or not isinstance(world,(int,float)) or not math.isfinite(world):
                    raise TickContractError(['Pause did not supply a valid native clock boundary'])
                if previous and world<previous['physics_time_s']:raise TickContractError(['Pause rewound the World clock'])
            baseline=(dict(epoch=epoch,sim_ns=stamp,physics_time_s=world,drive_targets=list(targets),kind='pause') if targets is not None else None)
        clock_anchor=dict(epoch=epoch,sim_ns=stamp,physics_time_s=world)
        if anchor:
            delta=stamp-anchor_stamp
            if not matches(delta,0,'native_ns') and not matches(delta,STEP_NS,'native_ns'):
                raise TickContractError(['Cancellation boundary skipped native frames'])
            old_world=anchor.get('physics_time_s')
            if old_world is not None:
                if world is None:raise TickContractError(['Cancellation boundary omitted its World clock'])
                errors=(world_step_errors(world,old_world) if delta>10 else
                        ([] if matches(world,old_world,'world_s') else ['Held cancellation boundary changed the World clock']))
                if errors:raise TickContractError(errors)
        self.previous=None;self.boundary=baseline;self.anchor=clock_anchor
        return copy.deepcopy(baseline)
    def observe(self,row):
        if not self.previous and self.boundary is not None:row['drive_baseline']=copy.deepcopy(self.boundary)
        errors=observation_errors(row,self.previous)
        if self.previous is None and self.anchor is not None:
            if row['epoch']!=self.anchor['epoch']:errors.append('First tick belongs to another boundary epoch')
            if not matches(row['pose_stamp_ns']-self.anchor['sim_ns'],STEP_NS,'native_ns'):errors.append('First tick skipped the retained clock anchor')
            if self.anchor.get('physics_time_s') is not None:
                errors.extend(world_step_errors(row['physics_time_s'],self.anchor['physics_time_s']))
        if (not self.previous or self.previous['epoch']!=row['epoch']) and not matches(row['pose_stamp_ns']-row['invalid_through_ns'],STEP_NS,'native_ns'):
            errors.append('First physical tick skipped its reset boundary')
        if errors:raise TickContractError(errors,dict(row=copy.deepcopy(row),comparisons=row_comparisons(row,self.previous)))
    def finish(self,row):
        errors=row_errors(row,self.previous)
        if errors:raise TickContractError(errors,dict(row=copy.deepcopy(row),comparisons=row_comparisons(row,self.previous)))
        proof=dict(version=1,passed=True,row=copy.deepcopy(row),previous=copy.deepcopy(self.previous),row_sha256=digest(row))
        # Previous is one row, not a recursively growing proof chain.
        self.previous=copy.deepcopy(row)
        return proof


def verify_proof(proof):
    try:
        row=proof['row'];errors=row_errors(row,proof['previous'])
        if proof.get('version')!=1 or not proof.get('passed') or proof['row_sha256']!=digest(row):errors.append('Tick proof identity differs')
        return errors
    except (KeyError,TypeError,ValueError):return ['Incomplete tick proof']



def row_comparisons(row,previous=None):
    stamp=row.get('sim_ns');records=[]
    for key in ('joint_stamp_ns','wrist_stamp_ns','top_stamp_ns','pose_stamp_ns'):
        records.append(comparison(key,row.get(key),stamp,'stamp_ns'))
    for key in ('expert_observation_stamp_ns','observation_stamp_ns','declared_application_stamp_ns'):
        records.append(comparison(key,row.get(key),stamp))
    for a,b,rule in (('state','physics_state','state_rad'),('velocity','physics_velocity','state_rad'),
                     ('executed_action','applied_command','target_rad')):
        records.append(comparison(a+' vs '+b,row.get(a),row.get(b),rule))
    for key in ('application_stamp_ns','command_clock_ns'):
        value=row.get(key)
        delta=value-stamp if isinstance(value,(int,float)) and isinstance(stamp,(int,float)) else None
        records.append(bound_comparison(key+' latency',delta,0,STEP_NS,'stamp_ns'))
    if previous:
        current=row.get('physics_time_s');old=previous.get('physics_time_s')
        if all(isinstance(v,(int,float)) and math.isfinite(v) for v in (current,old)):
            tolerance=max(1e-6,8*math.ulp(max(abs(current),abs(old),1.))+PHYSICS_STEPS*abs(FLOAT32_DT-PHYSICS_DT))
            record=comparison('physics_time_s progression',current-old,PHYSICS_STEPS*PHYSICS_DT,'world_s',tolerance=tolerance)
            record.update(source_left=current,source_right=old,expected_delta_s=PHYSICS_STEPS*PHYSICS_DT)
            record['passed']=record['passed'] and current>old and tolerance<PHYSICS_DT/4
            records.append(record)
        records.append(comparison('native pose_stamp_ns step',row['pose_stamp_ns']-previous['pose_stamp_ns'],STEP_NS,'native_ns'))
        records.append(comparison('physics_drive_targets vs prior ACK',row.get('physics_drive_targets'),previous.get('applied_command'),'target_rad'))
    if previous is not None:
        for record in row_comparisons(previous):
            records.append(dict(record,field='previous.'+record['field']))
    return [record for record in records if not record['passed']]


def snapshot_contracts(snapshots):
    errors=[];proofs=[];comparisons=[];last=None
    if not snapshots:errors.append('No physical verdict ticks')
    for snapshot in snapshots:
        proof=snapshot.get('tick_contract',{});errors.extend(verify_proof(proof));proofs.append(proof)
        if not proof.get('row'):continue
        row=proof['row']
        comparisons.extend(row_comparisons(row,proof.get('previous')))
        # Native fields are exact copies. Camera/ROS producer agreement is
        # separately checked using the same rules as observation_errors.
        fields=(('epoch','epoch','copy'),('sim_ns','pose_stamp_ns','copy'),
                ('sim_ns','sim_ns','stamp_ns'),('state','physics_state','copy'),
                ('state','state','state_rad'),('velocity','physics_velocity','copy'),
                ('velocity','velocity','state_rad'),('state_source','state_source','copy'),
                ('physics_time_s','physics_time_s','copy'),('bodies','physical_bodies','copy'))
        for source,target,rule in fields:
            record=comparison('snapshot.'+source+' vs row.'+target,snapshot.get(source),row.get(target),rule)
            if not record['passed']:
                comparisons.append(record)
                errors.append('Verdict snapshot differs from its checked tick: '+record['field'])
        if last:errors.extend(row_errors(row,last['tick_contract']['row']))
        if last and (snapshot['epoch']!=last['epoch'] or not matches(snapshot['sim_ns']-last['sim_ns'],STEP_NS,'native_ns')):
            errors.append('Verdict window has an epoch or clock gap')
        last=snapshot
    return dict(passed=not errors,errors=errors,proofs=proofs,comparisons=comparisons)


def require_snapshots(snapshots):
    contract=snapshot_contracts(snapshots)
    if not contract['passed']:raise TickContractError(contract['errors'],dict(contract=contract,snapshots=copy.deepcopy(snapshots)))
    return contract


def verdict_contract_valid(contract):
    if not contract or not contract.get('passed') or not contract.get('proofs'):return False
    previous=None
    for proof in contract['proofs']:
        if verify_proof(proof):return False
        if previous and (proof['row']['epoch']!=previous['epoch'] or row_errors(proof['row'],previous)):return False
        previous=proof['row']
    return True
