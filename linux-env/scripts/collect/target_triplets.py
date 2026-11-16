"""Counterfactual RGB episodes from one sealed layout with explicit pose agreement."""
import argparse,copy,hashlib,itertools,json,subprocess
from pathlib import Path
from contract import COLORS,parameters,COUNTERFACTUAL_TRAIN_SEEDS,COUNTERFACTUAL_HELDOUT_SEEDS,VISIBLE_TARGET_RULE,apply_target_rule
from comparisons import matches,interval
from scene_identity import physical_state,identity,require_identical,require_pose_agreement,prior_agreements,TRIPLET_POSE_POLICY,require_observation_drift,TRIPLET_OBSERVATION_DRIFT_POLICY,verify_pose_evidence
from tick_contract import STEP_NS
from storage import atomic_json,CANONICAL_ROOT,confined
import triplet_boundary as boundary
from tick_contract import digest

TRAIN_SEEDS=COUNTERFACTUAL_TRAIN_SEEDS
HELDOUT_SEEDS=COUNTERFACTUAL_HELDOUT_SEEDS
POLICY=dict(version=6,kind='counterfactual-target-triplet',colors=list(COLORS),normal_only=True,
    order='six counterbalanced permutations by layout ordinal',
    target_rule=dict(name=VISIBLE_TARGET_RULE,frame='/World/base',refuse_margin_le_m=.005),camera_prime_ticks=20,
    initial_observation_ticks=1,state_identity='soarm101-native-physical-state-v1',pose_agreement=copy.deepcopy(TRIPLET_POSE_POLICY),
    observation_anchor_drift=copy.deepcopy(TRIPLET_OBSERVATION_DRIFT_POLICY),
    splits=dict(train=list(TRAIN_SEEDS),heldout=list(HELDOUT_SEEDS)))


def spec(seed,split):
    seeds=TRAIN_SEEDS if split=='train' else HELDOUT_SEEDS if split=='heldout' else ()
    if seed not in seeds:raise ValueError('Layout seed is not reserved for this triplet split')
    order=list(tuple(itertools.permutations(COLORS))[(seed-seeds[0])%6])
    return dict(policy=copy.deepcopy(POLICY),layout_id='layout-'+str(seed),layout_seed=seed,split=split,color_order=order)


def member(config,index,snapshot=None):
    if index not in range(3):raise ValueError('A target triplet has exactly three members')
    value=parameters(config['layout_seed'],config['color_order'][index],split=config['split'],regrasp=False)
    value.update(trial=index)
    return apply_target_rule(value,snapshot) if snapshot is not None else value


def destination(root,config,base=CANONICAL_ROOT):
    expected=Path(base)/'datasets/counterfactual-targets'/config['split']/config['layout_id']
    if Path(root).resolve()!=expected.resolve():raise ValueError('Triplet output must use its separate split/layout directory')
    return expected


def configure_args(args,parser):
    seed=getattr(args,'counterfactual_layout_seed',None)
    if seed is None:
        if getattr(args,'counterfactual_split','train')!='train':parser.error('Counterfactual split needs a layout seed')
        if getattr(args,'seed',None) in TRAIN_SEEDS+HELDOUT_SEEDS:parser.error('Reserved triplet seed requires explicit counterfactual mode')
        try:require_mode(args.output,{})
        except ValueError as error:parser.error(str(error))
        return None
    if args.resume or args.replay_receipt or args.scene_bank or args.regrasp_only:
        parser.error('Counterfactual triplets require a fresh normal run; no replay, resume or scene-only bank')
    try:
        config=spec(seed,args.counterfactual_split);destination(confined(args.output),config)
    except ValueError as error:parser.error(str(error))
    args.normal_only=True;args.episodes=3;args.max_attempts=3;args.seed=seed
    return config


def restore_code(snapshot,triplet):
    return 'import collection_runtime as c\nresult=c.runtime().restore('+repr(snapshot)+',arm=True,restore_drives=True,triplet_state='+repr(triplet)+')'


class TripletSession:
    def __init__(self,root,config):
        if config!=spec(config['layout_seed'],config['split']):raise ValueError('Triplet configuration differs from the fixed recipe')
        self.root=Path(root);self.config=copy.deepcopy(config);self.anchor=None;self.bootstrap=None
        self.cup_sampler=None;self.sampled=0
        self.starts={};self.recorded_starts=set()
        self.prime_members=[];self.restore_members=[];self.bootstrap_members=[];self.first_members=[];self.first_deviations={};self.first_reference_deviations={}
        self.first_anchor_deviations={}
        if (self.root/'triplet-anchor.json').exists():raise ValueError('A triplet anchor cannot be silently adopted')
    def save(self):
        atomic_json(self.root/'triplet-anchor.json',dict(config=self.config,anchor=self.anchor,
            anchor_sha256=identity(self.anchor),bootstrap=self.bootstrap,
            bootstrap_sha256=identity(self.bootstrap) if self.bootstrap is not None else None,
            target_selections={member(self.config,i,self.anchor)['color']:member(self.config,i,self.anchor)['target_selection'] for i in range(3)},
            cup_sampler=self.cup_sampler,sampler_invocations=self.sampled,triplet_id=self.triplet_id))
    def prepare(self,c,params,baseline,receipt,motion_epochs,root,path):
        index=params['trial']
        if params!=member(self.config,index):raise ValueError('Triplet member parameters differ')
        if c.writer is not None:raise RuntimeError('Triplet preparation requires a closed writer')
        if self.anchor is None:
            if index!=0:raise ValueError('Triplet must sample its first member before the others')
            c._prepare_normal_scene_attempt(params,baseline,receipt,motion_epochs,root,path)
            current=c.held_snapshot(c.generation.epoch)
            canonical=c.restore_physical(restore_code(current,False))
            self.anchor=copy.deepcopy(canonical['snapshot']);physical_state(self.anchor)
            self.cup_sampler=copy.deepcopy(receipt['cup_randomization']);self.sampled=1
            self.triplet_id=hashlib.sha256(json.dumps(dict(config=self.config,state=identity(self.anchor)),sort_keys=True).encode()).hexdigest()
            self.save()
        params.update(member(self.config,index,self.anchor))
        c.timeline(False)
        cleared=c.scene_reset.clear()
        receipt['gpu_isolation_before']=c.gpu_footprint();receipt['camera_graph']=c.camera_graph()
        primed=c.restore_physical(restore_code(self.anchor,True))
        boundary.restore_admission(primed['snapshot'],primed,self.anchor)
        prime_prior=boundary.restore_prior(primed,self.prime_members)
        c.servo.configure(params)
        c.advance_ticks(POLICY['camera_prime_ticks'])
        restored=c.restore_physical(restore_code(self.anchor,True))
        native_proof=boundary.restore_admission(restored['snapshot'],restored,self.anchor)
        proof=native_proof['against_target'];restore_prior=boundary.restore_prior(restored,self.restore_members)
        observed=c.advance_ticks(POLICY['initial_observation_ticks'])[-1]
        boundary.bootstrap(restored['snapshot'],{k:v for k,v in restored.items() if k!='snapshot'},observed)
        observed_rule=boundary.selections(observed,self.anchor)
        bootstrap_anchor_drift=require_observation_drift(observed,self.anchor)
        if self.bootstrap is None:self.bootstrap=copy.deepcopy(observed);self.save()
        bootstrap_proof=require_pose_agreement(observed,self.bootstrap)
        bootstrap_prior=prior_agreements(observed,self.bootstrap_members)
        c.sync_scene(snapshot=observed);c.update_pose_cache(observed)
        selected=[member(self.config,i,self.anchor)['object'] for i in range(3)]
        visibility=c.mutate_kit('run_python',{'code':'from so_arm101_dt.extension import get_instance\ni=get_instance()\nposes='+repr(observed['bodies'])+'\nresult={name:bool(i._is_in_camera_view(poses[name]["position"],margin_px=30)) for name in '+repr(selected)+'}'})
        if set(visibility)!=set(selected) or not all(v is True for v in visibility.values()):raise RuntimeError('Every deterministically selected triplet target must be visible')
        c.servo.configure(params)
        with c.generation.own('counterfactual initial reference'):
            c.generation.require(observed['epoch'])
            c.servo.reference=list(observed['drive_targets']);c.servo.reference_epoch=observed['epoch']
        c.episode_start_physical=observed;c.scene_restore=restored
        c.node.approach_offset=params['approach_offset_m'];c.node._grasp_approach_height_var.set(params['approach_height_m']);c.node.speed_scale=params['speed_scale']
        motion_epochs.add(observed['epoch'])
        self.starts[receipt['episode_id']]=copy.deepcopy(observed)
        self.prime_members.append(copy.deepcopy(primed));self.restore_members.append(copy.deepcopy(restored));self.bootstrap_members.append(copy.deepcopy(observed))
        receipt.update(cup_randomization=copy.deepcopy(self.cup_sampler),initial_objects=dict(c.node.objects_data),initial_cups=dict(c.node._drop_data),
            initial_physical=restored['snapshot'],episode_start_physical=observed,initial_arm_check=dict(passed=True,exact_restore=proof['exact_bits'],pose_restore=proof),
            initial_restore_check={k:v for k,v in restored.items() if k!='snapshot'},initial_restore_error_m=restored['max_position_error_m'],
            initial_expert_reference=list(c.servo.reference),planning_scene_pose_check=c.scene_pose_check,planning_scene_clear_check=c.scene_clear_check,
            camera_duplicate_messages_before_episode=dict(c.sensors.duplicate_messages),
            counterfactual=dict(layout_id=self.config['layout_id'],triplet_id=self.triplet_id,layout_seed=self.config['layout_seed'],split=self.config['split'],
                anchor_sha256=identity(self.anchor),initial_observation_sha256=identity(observed),restore_proof=proof,
                initial_observation_proof=bootstrap_proof,visibility=visibility,sampler_invocations=self.sampled,
                priming_restore=primed,priming_prior=prime_prior,restore_prior=restore_prior,bootstrap_prior=bootstrap_prior,
                observed_target_rule=observed_rule,bootstrap_anchor_drift=bootstrap_anchor_drift,
                camera_prime_ticks=POLICY['camera_prime_ticks'],scene_clear_before_restore=cleared))

    def record_start(self,snapshot,row,path):
        episode=row['episode_id']
        if episode in self.recorded_starts:return
        record=boundary.recorded(snapshot,row,self.starts[episode],self.anchor,episode,self.triplet_id)
        deviations=prior_agreements(snapshot,self.first_members)
        reference_deviation=require_pose_agreement(snapshot,self.first_members[0] if self.first_members else snapshot)
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        if (path/'first-recorded-native.json').exists():raise ValueError('Cannot replace a native start certificate')
        atomic_json(path/'first-recorded-native.json',record);self.recorded_starts.add(episode)
        self.first_deviations[episode]=deviations;self.first_members.append(copy.deepcopy(snapshot))
        self.first_reference_deviations[episode]=reference_deviation
        self.first_anchor_deviations[episode]=record['anchor_drift']


def validate(config,anchor,receipts):
    if config!=spec(config['layout_seed'],config['split']):raise ValueError('Triplet recipe differs')
    if len(receipts)!=3 or not all(r.get('accepted') is True for r in receipts):raise ValueError('Only a complete successful RGB triplet may be exported')
    if anchor['config']!=config or anchor['sampler_invocations']!=1:raise ValueError('Triplet anchor configuration/sampling differs')
    if identity(anchor['anchor'])!=anchor['anchor_sha256'] or identity(anchor['bootstrap'])!=anchor['bootstrap_sha256']:raise ValueError('Triplet physical anchor changed')
    triplet_id=hashlib.sha256(json.dumps(dict(config=config,state=anchor['anchor_sha256']),sort_keys=True).encode()).hexdigest()
    if anchor['triplet_id']!=triplet_id:raise ValueError('Triplet identity differs from its anchor')
    selections={member(config,i,anchor['anchor'])['color']:member(config,i,anchor['anchor'])['target_selection'] for i in range(3)}
    verify_pose_evidence(anchor['target_selections'],selections,'Triplet visible target selections differ')
    if len({r['episode_id'] for r in receipts})!=3:raise ValueError('Triplet episode identities must be unique')
    for index,r in enumerate(receipts):
        verify_pose_evidence(r['params'],member(config,index,anchor['anchor']),'Triplet task/target/common parameters differ')
        m=r['counterfactual']
        for key,value in dict(layout_id=config['layout_id'],layout_seed=config['layout_seed'],split=config['split'],triplet_id=triplet_id,
                anchor_sha256=anchor['anchor_sha256'],initial_observation_sha256=identity(r['episode_start_physical']),sampler_invocations=1,camera_prime_ticks=POLICY['camera_prime_ticks']).items():
            if m[key]!=value:raise ValueError('Triplet metadata differs: '+key)
        proof=require_pose_agreement(r['initial_physical'],anchor['anchor']);bootstrap=require_pose_agreement(r['episode_start_physical'],anchor['bootstrap'])
        verify_pose_evidence([m['restore_proof'],m['initial_observation_proof']],[proof,bootstrap],'Triplet restore proof differs')
        restored=r['initial_physical'];observed=r['episode_start_physical']
        verify_pose_evidence(m['bootstrap_anchor_drift'],require_observation_drift(observed,anchor['anchor']),'Bootstrap anchor drift evidence differs')
        boundary.restore_admission(restored,r['initial_restore_check'],anchor['anchor'])
        prime=m['priming_restore'];boundary.restore_admission(prime['snapshot'],prime,anchor['anchor'])
        verify_pose_evidence([m['priming_prior'],m['restore_prior'],m['bootstrap_prior']],
            [boundary.restore_prior(prime,[p['counterfactual']['priming_restore'] for p in receipts[:index]]),
             boundary.restore_prior(dict(r['initial_restore_check'],snapshot=restored),[dict(p['initial_restore_check'],snapshot=p['initial_physical']) for p in receipts[:index]]),
             prior_agreements(observed,[p['episode_start_physical'] for p in receipts[:index]])],
            'Cross-member restore/bootstrap pose agreement evidence differs')
        boundary.bootstrap(restored,r['initial_restore_check'],observed)
        boundary.later(restored,receipts[index-1]['episode_start_physical'] if index else anchor['anchor'])
        boot_row=observed['tick_contract']['row']
        if (boot_row['episode_id']!=r['episode_id'] or boot_row['layout_id']!=config['layout_id'] or boot_row['triplet_id']!=triplet_id):
            raise ValueError('Bootstrap native proof belongs to another member/group')
        if index==0 and boundary.clocks(observed)!=boundary.clocks(anchor['bootstrap']):raise ValueError('Stored bootstrap native identity differs from first member')
        if index==0:require_identical(observed,anchor['bootstrap'])
        verify_pose_evidence(m['observed_target_rule'],boundary.selections(observed,anchor['anchor']),'Observed visible target rule evidence differs')
        visible={member(config,i,anchor['anchor'])['object'] for i in range(3)}
        if r['cup_randomization']!=anchor['cup_sampler'] or set(m['visibility'])!=visible or not all(v is True for v in m['visibility'].values()):raise ValueError('Triplet scene/visibility differs')
    return dict(version=1,complete=True,layout_id=config['layout_id'],triplet_id=triplet_id,layout_seed=config['layout_seed'],split=config['split'],
        anchor_sha256=anchor['anchor_sha256'],initial_observation_sha256=anchor['bootstrap_sha256'],colors=list(config['color_order']),
        target_selections=selections,episodes=[dict(episode_index=i,episode_id=r['episode_id'],task=r['params']['task'],target=r['params']['object']) for i,r in enumerate(receipts)])


def require_mode(run,document):
    """A lost field must not reinterpret a triplet as a normal run."""
    run=Path(run);config=document.get('counterfactual_triplet')
    if config is not None:return config
    marker=('counterfactual_triplet' in document or 'counterfactual-targets' in run.resolve().parts or
        any((run/name).exists() or (run/name).is_symlink() for name in ('triplet-anchor.json','counterfactual-triplet.json','dataset/meta/counterfactual_triplets.json')))
    summary=run/'summary.json'
    if summary.exists() and 'counterfactual_triplet_seal' in json.loads(summary.read_text()):marker=True
    for path in run.glob('attempt-*/receipt.json'):
        r=json.loads(path.read_text());p=r.get('params',{})
        if 'counterfactual' in r or 'target_selection' in p or p.get('seed') in TRAIN_SEEDS+HELDOUT_SEEDS:marker=True
    if any(run.glob('attempt-*/first-recorded-native.json')):marker=True
    if not marker:
        for path in run.glob('attempt-*/frames.jsonl'):
            with path.open() as stream:
                if any('layout_id' in r or 'triplet_id' in r for r in map(json.loads,stream)):marker=True;break
    if marker:raise ValueError('Triplet artifacts require a valid counterfactual run configuration')
    return None


def seal_identity(run):
    paths=['run.json','triplet-anchor.json','counterfactual-triplet.json']
    paths += [f'attempt-{i:04d}/{name}' for i in range(3) for name in ('receipt.json','frames.jsonl','first-recorded-native.json')]
    return dict(version=1,files_sha256={p:hashlib.sha256((run/p).read_bytes()).hexdigest() for p in paths})


def seal_run(run,base=CANONICAL_ROOT):
    run=Path(run);result=validate_run(run,base,sealed=False)
    if result is None:raise ValueError('Cannot seal a normal run as a triplet')
    atomic_json(run/'counterfactual-triplet.json',result)
    return seal_identity(run)


def validate_run(run,base=CANONICAL_ROOT,sealed=True):
    run=Path(run);config=require_mode(run,json.loads((run/'run.json').read_text()))
    if config is None:return None
    destination(run,config,base);anchor=json.loads((run/'triplet-anchor.json').read_text())
    receipts=[json.loads(p.read_text()) for p in sorted(run.glob('attempt-*/receipt.json'))]
    result=validate(config,anchor,receipts);first=[];offsets=[];native_proofs=[];previous_last=None
    for index,r in enumerate(receipts):
        row=None;last=None
        with (run/f'attempt-{index:04d}/frames.jsonl').open() as stream:
            for line in stream:
                current=json.loads(line)
                if current['episode_id']!=r['episode_id'] or current['layout_id']!=result['layout_id'] or current['triplet_id']!=result['triplet_id']:raise ValueError('Recorded frame belongs to another triplet')
                if row is None:row=current
                last=current
        if row is None:raise ValueError('Triplet member has no recorded frames')
        if previous_last is not None:boundary.later(r['initial_physical'],previous_last)
        previous_last=boundary.from_row(last)
        record=json.loads((run/f'attempt-{index:04d}/first-recorded-native.json').read_text())
        boundary.verify_record(record,row,r['episode_start_physical'],anchor['anchor'],r['episode_id'],result['triplet_id'])
        verify_pose_evidence(r['counterfactual']['first_recorded_anchor_drift'],require_observation_drift(record['native'],anchor['anchor']),'First recorded anchor drift evidence differs')
        deviations=prior_agreements(record['native'],first)
        verify_pose_evidence(r['counterfactual']['first_recorded_prior'],deviations,'First recorded cross-member pose agreement evidence differs')
        reference_deviation=require_pose_agreement(record['native'],first[0] if first else record['native'])
        verify_pose_evidence(r['counterfactual']['first_recorded_deviation'],reference_deviation,'First recorded reference deviation evidence differs')
        native_proofs.append(dict(episode_id=r['episode_id'],certificate_sha256=digest(record),observed_rule=record['observed_rule']))
        offset=row['pose_stamp_ns']-r['episode_start_physical']['sim_ns']
        offsets.append(offset)
        first.append(dict(state=row['physics_state'],velocity=row['physics_velocity'],drive_targets=row['physics_drive_targets'],
            bodies=row['physical_bodies'],base_world_pose=row['base_world_pose'],robot_base_world_pose=row['robot_base_world_pose']))
    if any(not matches(offsets[0],v,'native_ns') for v in offsets):raise ValueError('Triplet first observation offsets differ')
    result.update(pose_agreement=copy.deepcopy(TRIPLET_POSE_POLICY),observation_anchor_drift=copy.deepcopy(TRIPLET_OBSERVATION_DRIFT_POLICY),first_recorded_physical_sha256=[identity(v) for v in first],
        member_deviations=[dict(episode_id=r['episode_id'],restore=r['counterfactual']['restore_proof'],
            restore_prior=r['counterfactual']['restore_prior'],priming_prior=r['counterfactual']['priming_prior'],
            bootstrap=r['counterfactual']['initial_observation_proof'],bootstrap_prior=r['counterfactual']['bootstrap_prior'],
            anchor_drift=dict(bootstrap=r['counterfactual']['bootstrap_anchor_drift'],first_recorded=r['counterfactual']['first_recorded_anchor_drift']),
            first_recorded=r['counterfactual']['first_recorded_deviation'],first_recorded_prior=r['counterfactual']['first_recorded_prior'],reread=r['initial_restore_check']['triplet_physical_state']) for r in receipts],
        first_recorded_offsets_ns=offsets,native_start_proofs=native_proofs)
    if sealed:
        verify_pose_evidence(json.loads((run/'counterfactual-triplet.json').read_text()),result,'Stored triplet group certificate differs')
        if json.loads((run/'summary.json').read_text()).get('counterfactual_triplet_seal')!=seal_identity(run):raise ValueError('Finalized triplet inputs/configuration changed')
    return result


def plan(base,instances,train_layouts=120,heldout_layouts=20):
    if not instances or len(set(instances))!=len(instances) or any(i not in range(10) for i in instances):raise ValueError('Unique instance indices0–9 required')
    if not 0<=train_layouts<=120 or not 0<=heldout_layouts<=20:raise ValueError('Layout count exceeds reserved plan')
    jobs=[]
    for split,seeds in [('train',TRAIN_SEEDS[:train_layouts]),('heldout',HELDOUT_SEEDS[:heldout_layouts])]:
        for seed in seeds:
            config=spec(seed,split);index=instances[len(jobs)%len(instances)];output=Path(base)/'datasets/counterfactual-targets'/split/config['layout_id']
            arguments=['--counterfactual-layout-seed',str(seed),'--counterfactual-split',split,'--normal-only','--output',str(output)]
            jobs.append(dict(config=config,instance=index,output=str(output),arguments=arguments,
                command=['bash','linux-env/scripts/collect_sim.sh','run',str(index),*arguments],
                export_command=['bash','linux-env/scripts/collect_sim.sh','export',str(index),str(output),
                    '--repo-id','local/soarm101_counterfactual_'+split,'--report-dir',str(output/'report')]))
    return dict(version=1,cpu_plan_only=True,production_authorized=False,episodes=3*len(jobs),training_episodes=3*train_layouts,heldout_episodes=3*heldout_layouts,jobs=jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--target-host',choices=('dual-a4500','rtx6000'),default='rtx6000')
    p.add_argument('--instances',default='3,4,5,6');p.add_argument('--train-layouts',type=int,default=120);p.add_argument('--heldout-layouts',type=int,default=20)
    a=p.parse_args();profile=json.loads((Path(__file__).with_name('hosts')/(a.target_host+'.json')).read_text())
    value=plan(profile['root'],list(map(int,a.instances.split(','))),a.train_layouts,a.heldout_layouts);out=confined(a.output)
    repo=Path(__file__).resolve().parents[3]
    value['source_commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    value['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('target_triplets.py','triplet_boundary.py','scene_identity.py','collect.py','contract.py','collection_runtime.py','export.py','export_content.py','export_video.py')}
    if out.exists():raise FileExistsError('Plan already exists')
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,value);print(json.dumps(value))


if __name__=='__main__':main()
