"""Counterfactual RGB episodes from one sealed, exactly restored physical layout."""
import argparse,copy,hashlib,itertools,json,subprocess
from pathlib import Path
from contract import COLORS,parameters,COUNTERFACTUAL_TRAIN_SEEDS,COUNTERFACTUAL_HELDOUT_SEEDS,VISIBLE_TARGET_RULE,apply_target_rule
from comparisons import matches,interval
from scene_identity import physical_state,identity,require_identical
from tick_contract import STEP_NS
from storage import atomic_json,CANONICAL_ROOT,confined

TRAIN_SEEDS=COUNTERFACTUAL_TRAIN_SEEDS
HELDOUT_SEEDS=COUNTERFACTUAL_HELDOUT_SEEDS
POLICY=dict(version=2,kind='counterfactual-target-triplet',colors=list(COLORS),normal_only=True,
    order='six counterbalanced permutations by layout ordinal',
    target_rule=dict(name=VISIBLE_TARGET_RULE,frame='/World/base',refuse_margin_le_m=.005),camera_prime_ticks=20,
    initial_observation_ticks=1,state_identity='soarm101-native-physical-state-v1',
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
        return None
    if args.resume or args.replay_receipt or args.scene_bank or args.regrasp_only:
        parser.error('Counterfactual triplets require a fresh normal run; no replay, resume or scene-only bank')
    try:
        config=spec(seed,args.counterfactual_split);destination(confined(args.output),config)
    except ValueError as error:parser.error(str(error))
    args.normal_only=True;args.episodes=3;args.max_attempts=3;args.seed=seed
    return config


def restore_code(snapshot,exact):
    return 'import collection_runtime as c\nresult=c.runtime().restore('+repr(snapshot)+',arm=True,restore_drives=True,exact_state='+repr(exact)+')'


class TripletSession:
    def __init__(self,root,config):
        if config!=spec(config['layout_seed'],config['split']):raise ValueError('Triplet configuration differs from the fixed recipe')
        self.root=Path(root);self.config=copy.deepcopy(config);self.anchor=None;self.bootstrap=None
        self.cup_sampler=None;self.sampled=0
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
        require_identical(primed['snapshot'],self.anchor)
        c.servo.configure(params)
        c.advance_ticks(POLICY['camera_prime_ticks'])
        restored=c.restore_physical(restore_code(self.anchor,True))
        proof=require_identical(restored['snapshot'],self.anchor)
        observed=c.advance_ticks(POLICY['initial_observation_ticks'])[-1]
        if observed['epoch']!=restored['epoch'] or not matches(observed['sim_ns']-restored['snapshot']['sim_ns'],STEP_NS,'native_ns'):
            raise RuntimeError('Triplet initial observation crosses a reset epoch or skips a tick')
        if self.bootstrap is None:self.bootstrap=copy.deepcopy(observed);self.save()
        bootstrap_proof=require_identical(observed,self.bootstrap)
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
        receipt.update(cup_randomization=copy.deepcopy(self.cup_sampler),initial_objects=dict(c.node.objects_data),initial_cups=dict(c.node._drop_data),
            initial_physical=restored['snapshot'],episode_start_physical=observed,initial_arm_check=dict(passed=True,exact_restore=proof),
            initial_restore_check={k:v for k,v in restored.items() if k!='snapshot'},initial_restore_error_m=restored['max_position_error_m'],
            initial_expert_reference=list(c.servo.reference),planning_scene_pose_check=c.scene_pose_check,planning_scene_clear_check=c.scene_clear_check,
            camera_duplicate_messages_before_episode=dict(c.sensors.duplicate_messages),
            counterfactual=dict(layout_id=self.config['layout_id'],triplet_id=self.triplet_id,layout_seed=self.config['layout_seed'],split=self.config['split'],
                anchor_sha256=identity(self.anchor),initial_observation_sha256=identity(self.bootstrap),restore_proof=proof,
                initial_observation_proof=bootstrap_proof,visibility=visibility,sampler_invocations=self.sampled,
                camera_prime_ticks=POLICY['camera_prime_ticks'],scene_clear_before_restore=cleared))


def validate(config,anchor,receipts):
    if config!=spec(config['layout_seed'],config['split']):raise ValueError('Triplet recipe differs')
    if len(receipts)!=3 or not all(r.get('accepted') is True for r in receipts):raise ValueError('Only a complete successful RGB triplet may be exported')
    if anchor['config']!=config or anchor['sampler_invocations']!=1:raise ValueError('Triplet anchor configuration/sampling differs')
    if identity(anchor['anchor'])!=anchor['anchor_sha256'] or identity(anchor['bootstrap'])!=anchor['bootstrap_sha256']:raise ValueError('Triplet physical anchor changed')
    triplet_id=hashlib.sha256(json.dumps(dict(config=config,state=anchor['anchor_sha256']),sort_keys=True).encode()).hexdigest()
    if anchor['triplet_id']!=triplet_id:raise ValueError('Triplet identity differs from its anchor')
    selections={member(config,i,anchor['anchor'])['color']:member(config,i,anchor['anchor'])['target_selection'] for i in range(3)}
    if anchor['target_selections']!=selections:raise ValueError('Triplet visible target selections differ')
    if len({r['episode_id'] for r in receipts})!=3:raise ValueError('Triplet episode identities must be unique')
    for index,r in enumerate(receipts):
        if r['params']!=member(config,index,anchor['anchor']):raise ValueError('Triplet task/target/common parameters differ')
        m=r['counterfactual']
        for key,value in dict(layout_id=config['layout_id'],layout_seed=config['layout_seed'],split=config['split'],triplet_id=triplet_id,
                anchor_sha256=anchor['anchor_sha256'],initial_observation_sha256=anchor['bootstrap_sha256'],sampler_invocations=1,camera_prime_ticks=POLICY['camera_prime_ticks']).items():
            if m[key]!=value:raise ValueError('Triplet metadata differs: '+key)
        proof=require_identical(r['initial_physical'],anchor['anchor']);bootstrap=require_identical(r['episode_start_physical'],anchor['bootstrap'])
        if m['restore_proof']!=proof or m['initial_observation_proof']!=bootstrap:raise ValueError('Triplet restore proof differs')
        restored=r['initial_physical'];observed=r['episode_start_physical']
        if (restored['epoch']!=r['initial_restore_check']['epoch'] or observed['epoch']!=restored['epoch'] or
                not matches(observed['sim_ns']-restored['sim_ns'],STEP_NS,'native_ns')):raise ValueError('Triplet clock/epoch boundary differs')
        visible={member(config,i,anchor['anchor'])['object'] for i in range(3)}
        if r['cup_randomization']!=anchor['cup_sampler'] or set(m['visibility'])!=visible or not all(v is True for v in m['visibility'].values()):raise ValueError('Triplet scene/visibility differs')
    return dict(version=1,complete=True,layout_id=config['layout_id'],triplet_id=triplet_id,layout_seed=config['layout_seed'],split=config['split'],
        anchor_sha256=anchor['anchor_sha256'],initial_observation_sha256=anchor['bootstrap_sha256'],colors=list(config['color_order']),
        target_selections=selections,episodes=[dict(episode_index=i,episode_id=r['episode_id'],task=r['params']['task'],target=r['params']['object']) for i,r in enumerate(receipts)])


def validate_run(run,base=CANONICAL_ROOT):
    run=Path(run);config=json.loads((run/'run.json').read_text()).get('counterfactual_triplet')
    if config is None:return None
    destination(run,config,base);anchor=json.loads((run/'triplet-anchor.json').read_text())
    receipts=[json.loads(p.read_text()) for p in sorted(run.glob('attempt-*/receipt.json'))]
    result=validate(config,anchor,receipts);first=[];offsets=[]
    for index,r in enumerate(receipts):
        row=None
        with (run/f'attempt-{index:04d}/frames.jsonl').open() as stream:
            for line in stream:
                current=json.loads(line)
                if current['episode_id']!=r['episode_id'] or current['layout_id']!=result['layout_id'] or current['triplet_id']!=result['triplet_id']:raise ValueError('Recorded frame belongs to another triplet')
                if row is None:row=current
        if row is None:raise ValueError('Triplet member has no recorded frames')
        offset=row['pose_stamp_ns']-r['episode_start_physical']['sim_ns']
        if row['epoch']!=r['episode_start_physical']['epoch'] or offset<0 or not interval(offset,0,STEP_NS,'native_ns'):
            raise ValueError('First triplet frame crosses its preparation epoch or clock window')
        offsets.append(offset)
        first.append(dict(state=row['physics_state'],velocity=row['physics_velocity'],drive_targets=row['physics_drive_targets'],
            bodies=row['physical_bodies'],base_world_pose=row['base_world_pose'],robot_base_world_pose=row['robot_base_world_pose']))
    for observed in first:require_identical(observed,first[0])
    if any(not matches(offsets[0],v,'native_ns') for v in offsets):raise ValueError('Triplet first observation offsets differ')
    result.update(first_recorded_scene_sha256=identity(first[0]),first_recorded_offsets_ns=offsets);return result


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
        for name in ('target_triplets.py','scene_identity.py','collect.py','contract.py','collection_runtime.py','export.py','export_content.py','export_video.py')}
    if out.exists():raise FileExistsError('Plan already exists')
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,value);print(json.dumps(value))


if __name__=='__main__':main()
