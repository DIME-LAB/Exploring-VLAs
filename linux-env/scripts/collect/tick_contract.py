"""Pipeline checks for every physical tick, including unrecorded verdict ticks."""
import copy,hashlib,json,math

STEP_NS=round(1e9/30)


class TickContractError(RuntimeError):
    def __init__(self,errors,evidence=None):
        self.errors=list(errors);self.evidence=evidence
        super().__init__('Physical tick contract failed: '+'; '.join(self.errors))


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def vector(value,size):
    return isinstance(value,(list,tuple)) and len(value)==size and all(isinstance(v,(int,float)) and math.isfinite(v) for v in value)


def observation_errors(row,previous=None):
    errors=[]
    stamp=row.get('sim_ns',0);epoch=row.get('epoch')
    if not isinstance(stamp,int) or stamp<=0:return ['Missing positive native sim stamp']
    if epoch is None or any(row.get(k)!=epoch for k in ('observation_epoch','expert_epoch')):errors.append('Tick epoch mismatch')
    if stamp<=row.get('invalid_through_ns',stamp):errors.append('Tick is at/before reset boundary')
    for key in ('joint_stamp_ns','wrist_stamp_ns','top_stamp_ns','pose_stamp_ns'):
        if abs(row.get(key,-10000)-stamp)>1000:errors.append(key+' differs from sim tick')
    if row.get('state_source')!='isaac_articulation':errors.append('Tick state is not measured')
    for key in ('state','velocity','physics_state','physics_velocity','physics_drive_targets'):
        if not vector(row.get(key),6):errors.append('Invalid '+key)
    if not errors:
        for a,b in (('state','physics_state'),('velocity','physics_velocity')):
            if max(abs(x-y) for x,y in zip(row[a],row[b]))>1e-7:errors.append(a+' differs from physical snapshot')
    for camera in ('wrist','top'):
        layout=row.get('camera_layout',{}).get(camera,{})
        if (layout.get('encoding'),layout.get('width'),layout.get('height'))!=('rgb8',640,480) or layout.get('step',0)<1920 or layout.get('bytes')!=480*layout.get('step',0):errors.append('Invalid camera layout '+camera)
    physical=row.get('physics_time_s')
    if not isinstance(physical,(int,float)) or not math.isfinite(physical) or abs(physical*1e9-stamp)>1000:
        errors.append('Native physics time differs from camera tick')
    for name,body in row.get('physical_bodies',{}).items():
        if not all(vector(body.get(k),size) for k,size in (('position',3),('orientation',4),('velocity',6))):
            errors.append('Invalid physical body '+name)
        elif sum(v*v for v in body['orientation'])<1e-10:errors.append('Invalid body quaternion '+name)
    for key in ('wall_ns','joint_receive_ns','wrist_receive_ns','top_receive_ns'):
        if not isinstance(row.get(key),int) or row[key]<=0:errors.append('Missing wall receipt '+key)
    if previous and previous.get('epoch')==epoch:
        if abs(stamp-previous['sim_ns']-STEP_NS)>10:errors.append('Repeated or skipped physical tick')
        if isinstance(physical,(int,float)) and abs((physical-previous['physics_time_s'])*1e9-STEP_NS)>1000:
            errors.append('Physics did not advance one frame')
        for key in ('wall_ns','wrist_receive_ns','top_receive_ns'):
            if row.get(key,0)<=previous.get(key,0):errors.append('Non-monotonic '+key)
        if vector(row.get('physics_drive_targets'),6) and vector(previous.get('applied_command'),6):
            if max(abs(x-y) for x,y in zip(row['physics_drive_targets'],previous['applied_command']))>1e-6:
                errors.append('Previously ACKed targets did not persist')
    return errors


def row_errors(row,previous=None):
    errors=observation_errors(row,previous)
    stamp=row.get('sim_ns',0)
    for key in ('action','executed_action','expert_reference','applied_command'):
        if not vector(row.get(key),6):errors.append('Invalid '+key)
    if row.get('expert_observation_stamp_ns')!=stamp:errors.append('Expert uses another observation stamp')
    if row.get('observation_stamp_ns')!=stamp:errors.append('ACK uses another observation stamp')
    if abs(row.get('declared_application_stamp_ns',-1)-stamp)>10:errors.append('Command scheduled at another tick')
    latency=row.get('application_stamp_ns',-100)-stamp
    if not -10<=latency<=STEP_NS+10:errors.append('Application exceeds one-tick contract')
    if not -1000<=row.get('command_clock_ns',-10000)-stamp<=STEP_NS+1000:errors.append('Command clock differs from held tick')
    if not errors:
        if max(abs(x-y) for x,y in zip(row['executed_action'],row['applied_command']))>1e-6:errors.append('ACK target mismatch')
        clean=[q+max(-.035,min(.035,r-q)) for q,r in zip(row['state'],row['expert_reference'])]
        if max(abs(x-y) for x,y in zip(clean,row['action']))>1e-9:errors.append('Noisy or stale expert label')
    if row.get('tick_contract_version'):
        for key,size in (('planner_execution_reference',6),('dart_offset',6),('command_filter_delta',6),('command_velocity',5)):
            if not vector(row.get(key),size):errors.append('Invalid execution provenance '+key)
        if not errors:
            if any(abs(e-r-n)>1e-9 for e,r,n in zip(row['executed_action'],row['planner_execution_reference'],row['dart_offset'])):errors.append('Executed target differs from planner plus offset')
            if any(row['command_filter_delta']) or row['dart_offset'][5]!=0.:errors.append('Expert path or gripper was perturbed')
            velocity=[(q-p)*30 for q,p in zip(row['executed_action'][:5],row['physics_drive_targets'][:5])]
            if any(abs(a-b)>1e-4 for a,b in zip(velocity,row['command_velocity'])):errors.append('Command velocity provenance differs')
    geometry=row.get('execution_geometry',{})
    if not geometry.get('passed') or geometry.get('epoch')!=row.get('epoch') or geometry.get('sim_ns')!=stamp or geometry.get('target')!=row.get('executed_action'):
        errors.append('Final command geometry identity differs')
    return errors


class TickContract:
    def __init__(self):self.previous=None
    def observe(self,row):
        errors=observation_errors(row,self.previous)
        if (not self.previous or self.previous['epoch']!=row['epoch']) and abs(row['sim_ns']-row['invalid_through_ns']-STEP_NS)>1000:
            errors.append('First physical tick skipped its reset boundary')
        if errors:raise TickContractError(errors,copy.deepcopy(row))
    def finish(self,row):
        errors=row_errors(row,self.previous)
        if errors:raise TickContractError(errors,copy.deepcopy(row))
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


def snapshot_contracts(snapshots):
    errors=[];proofs=[];last=None
    if not snapshots:errors.append('No physical verdict ticks')
    for snapshot in snapshots:
        proof=snapshot.get('tick_contract',{});errors.extend(verify_proof(proof));proofs.append(proof)
        if not proof.get('row'):continue
        row=proof['row']
        if any(snapshot.get(k)!=row.get(k) for k in ('epoch','sim_ns','state','velocity','state_source','physics_time_s')):
            errors.append('Verdict snapshot differs from its checked tick')
        if snapshot.get('bodies')!=row.get('physical_bodies'):errors.append('Verdict poses differ from checked tick')
        if last:errors.extend(row_errors(row,last['tick_contract']['row']))
        if last and (snapshot['epoch']!=last['epoch'] or abs(snapshot['sim_ns']-last['sim_ns']-STEP_NS)>10):
            errors.append('Verdict window has an epoch or clock gap')
        last=snapshot
    return dict(passed=not errors,errors=errors,proofs=proofs)


def require_snapshots(snapshots):
    contract=snapshot_contracts(snapshots)
    if not contract['passed']:raise TickContractError(contract['errors'])
    return contract


def verdict_contract_valid(contract):
    if not contract or not contract.get('passed') or not contract.get('proofs'):return False
    previous=None
    for proof in contract['proofs']:
        if verify_proof(proof):return False
        if previous and (proof['row']['epoch']!=previous['epoch'] or row_errors(proof['row'],previous)):return False
        previous=proof['row']
    return True
