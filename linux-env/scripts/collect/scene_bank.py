"""Scene-only publication around the Collector's actual preparation path."""
import copy,hashlib,json,math,os,shutil,uuid
from pathlib import Path
from contract import COLORS,EVAL_SEEDS,parameters

NAMES={f'{c}_2x3_{i}' for c in COLORS for i in range(2)}|{f'cup_{c}' for c in COLORS}
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex);tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def body_poses(snapshot):
    bodies=snapshot['bodies'];assert set(bodies)==NAMES,'Unexpected physical body set'
    result={}
    for name,b in bodies.items():
        assert len(b['position'])==3 and len(b['orientation'])==4 and all(math.isfinite(v) for v in b['position']+b['orientation'])
        result[name]={k:list(b[k]) for k in ('position','orientation')}
    return result
def pose_errors(actual,expected):
    assert set(actual)==set(expected)==NAMES
    pos=max(abs(a-b) for name in NAMES for a,b in zip(actual[name]['position'],expected[name]['position']))
    quat=max(min(max(abs(a-b) for a,b in zip(actual[name]['orientation'],expected[name]['orientation'])),max(abs(a+b) for a,b in zip(actual[name]['orientation'],expected[name]['orientation']))) for name in NAMES)
    return dict(max_position_m=pos,max_quaternion_component=quat)
def configure_args(args,parser):
    if not args.scene_bank:return
    if args.resume or args.replay_receipt or args.regrasp_only:parser.error('Scene-bank generation requires a fresh normal scene-only run')
    if not args.scene_bank_reference_run or not args.scene_bank_output:parser.error('Scene bank requires reference run.json and a fresh destination')
    if args.scene_bank=='eval':
        if args.scene_bank_receipts:parser.error('Eval generation cannot use training receipts')
        args.seed=900000;args.episodes=len(EVAL_SEEDS);args.max_attempts=args.episodes
    else:
        if not args.scene_bank_receipts:parser.error('Golden scene generation requires explicit training receipts')
        args.episodes=len(args.scene_bank_receipts);args.max_attempts=args.episodes
    args.normal_only=True

class SceneBankSession:
    def __init__(self,args):
        self.args=args;self.root=Path(args.output);self.destination=Path(args.scene_bank_output).resolve()
        if not self.destination.is_relative_to('/mnt/a4500-backup/exploring-vlas/eval'):raise ValueError('Bank destination must stay on eval backup storage')
        if self.destination.exists():raise FileExistsError('Refusing to overwrite a sealed bank')
        self.destination.parent.mkdir(parents=True,exist_ok=True)
        if self.root.stat().st_dev!=self.destination.parent.stat().st_dev:raise ValueError('Bank draft and destination must share a filesystem for atomic publication')
        self.reference_path=Path(args.scene_bank_reference_run);self.reference=json.loads(self.reference_path.read_text())
        self.draft=self.root/'bank-draft';self.draft.mkdir();self.receipts=[]
        self.sources=[json.loads(Path(p).read_text()) for p in (args.scene_bank_receipts or [])]
        if args.scene_bank=='eval':self.plan=[parameters(seed,COLORS[seed%3],split='eval',regrasp=False) for seed in EVAL_SEEDS]
        else:
            assert all(r['accepted'] and r['params'].get('recovery')=='none' for r in self.sources)
            self.plan=[]
            for r in self.sources:
                original=r['params'];computed=parameters(original['seed'],original['color'],split='train',regrasp=False)
                assert computed=={k:v for k,v in original.items() if k!='trial'},'Training parameter recipe differs'
                self.plan.append(copy.deepcopy(original))
        assert len({p['seed'] for p in self.plan})==len(self.plan)
    def bind(self,baseline,cup_layout,snapshot,context):
        expected=self.reference['baseline'];assert set(expected)==set(baseline)
        error=max(abs(a-b) for n in baseline for a,b in zip(baseline[n],expected[n]))
        assert error<=1e-6,'Live baseline differs from the training collector baseline'
        assert cup_layout==self.reference['baseline_cup_layout'],'Live CUP_LAYOUT differs from training baseline'
        self.startup_snapshot=copy.deepcopy(snapshot)
        self.provenance=dict(format_version=2,purpose='held-out-collector-scenes' if self.args.scene_bank=='eval' else 'golden-training-resample',sampler='collect.Collector._prepare_scene_attempt -> scene -> randomize_scene',scene_bank_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference_run=str(self.reference_path),reference_run_sha256=hashlib.sha256(self.reference_path.read_bytes()).hexdigest(),baseline=baseline,baseline_cup_layout=cup_layout,baseline_max_error_m=error,baseline_body_file_role='first sampled layout; sampler reset baseline is recorded here',context=context)
        atomic(self.draft/'metadata/provenance.json',self.provenance)
    def record(self,receipt):
        p=receipt['params'];index=len(self.receipts);assert p['seed']==self.plan[index]['seed'] and p['color']==self.plan[index]['color']
        assert receipt['cup_randomization']['status']=='success' and receipt['retirement']['passed'] and receipt['camera_graph']['passed'] and receipt['camera_boundary']['passed']
        for key in ('gpu_isolation_before','gpu_isolation_after'):assert receipt[key]['passed']
        visible=receipt['bank_visibility'];assert len(visible[p['color']])>=2,'Requested-colour visibility differs from collector'
        body=body_poses(receipt['initial_physical'])
        if not self.receipts:atomic(self.draft/'baseline.json',body)
        if self.sources:
            receipt['golden_cup_sampler_exact']=receipt['cup_randomization']==self.sources[index]['cup_randomization']
            assert receipt['golden_cup_sampler_exact'],'Golden deterministic cup sampler differs from training receipt'
            check=pose_errors(body,body_poses(self.sources[index]['initial_physical']));receipt['golden_pose_parity']=check
            assert check['max_position_m']<=1e-4 and check['max_quaternion_component']<=1e-4,'Golden scene resample differs from recorded training layout'
        atomic(self.draft/f"seed-{p['seed']}.json",body);atomic(self.draft/f"metadata/seed-{p['seed']}.json",receipt);self.receipts.append(copy.deepcopy(receipt))
        atomic(self.root/'bank-progress.json',dict(complete=False,layouts=len(self.receipts),planned=len(self.plan),seeds=[r['params']['seed'] for r in self.receipts]))
    def seal(self):
        assert len(self.receipts)==len(self.plan),'A partial scene bank cannot be sealed'
        from collection_status import require_finalized
        certificate=require_finalized(self.root)
        assert json.loads((self.root/'summary.json').read_text())['complete'] is True
        atomic(self.draft/'metadata/finalization.json',certificate)
        files=[self.draft/'baseline.json']+[self.draft/f"seed-{p['seed']}.json" for p in self.plan]
        metadata=list((self.draft/'metadata').glob('*.json'))
        manifest=dict(format_version=2,purpose=self.provenance['purpose'],seeds=[p['seed'] for p in self.plan],colors=list(COLORS),files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},visible={str(r['params']['seed']):r['bank_visibility'] for r in self.receipts},sampler_metadata_sha256={str(p.relative_to(self.draft)):hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata},parameters={str(r['params']['seed']):r['params'] for r in self.receipts},producer='collect.Collector.run_scene_bank',complete=True)
        atomic(self.draft/'bank-manifest.json',manifest)
        self.destination.parent.mkdir(parents=True,exist_ok=True)
        if self.destination.exists():raise FileExistsError('Bank destination appeared during generation')
        os.rename(self.draft,self.destination)
        proof=dict(complete=True,layouts=len(self.plan),bank=str(self.destination),manifest_sha256=hashlib.sha256((self.destination/'bank-manifest.json').read_bytes()).hexdigest());atomic(self.root/'bank-complete.json',proof);return proof
