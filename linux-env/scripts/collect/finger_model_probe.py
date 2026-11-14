"""Offline USD reader check; anonymous opinions, no Kit app or asset saves.

Run with Isaac's Python (pxr available). PhysxSchema offsets are intentionally
not emulated: the live smoke must read those from actual PhysX shapes.
"""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics
repo=Path(__file__).resolve().parents[3];sys.path[:0]=[str(repo/'linux-env/scripts/collect'),str(repo/'vla_SO-ARM101/src/so_arm101_control')]
from finger_geometry import read_model,FingerModel
from miss_clearance import POLICY
asset=repo/'isaac/exts/soarm101-dt/assets/SO-ARM101-USD.usd';stage=Usd.Stage.Open(str(asset));stage.SetEditTarget(stage.GetSessionLayer());root=stage.GetDefaultPrim()
path=repo/'isaac/exts/soarm101-dt/so_arm101_dt/primitive_colliders.py'
source=ast.parse(path.read_text())
functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('robot','box','disable_mesh_collisions')]
def collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    return prim  # Geometry probe only: PhysxSchema offsets need the live Kit readback.
namespace=dict(Gf=Gf,Usd=Usd,UsdGeom=UsdGeom,UsdPhysics=UsdPhysics,collision=collision)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
while True:
    instanced=[p for p in Usd.PrimRange(root) if p.IsInstanceable()]
    if not instanced:break
    for prim in instanced:prim.SetInstanceable(False)
namespace['robot'](stage,root)  # Anonymous session, no physics app or asset save.
model=read_model(stage,str(root.GetPath()));actual=FingerModel(model)
fixture=json.loads((repo/'linux-env/scripts/collect/fixtures/descent-obstructions.json').read_text());retained=FingerModel(fixture)
def key(path):return tuple(path.split('/')[2:])
errors={k:[] for k in ('center_world','rotation','half_extents_m')}
for seed in fixture['seeds']:
 a={key(b['path']):b for b in actual.boxes(seed['command'])};b={key(b['path']):b for b in retained.boxes(seed['command'])}
 if set(a)!=set(b):raise RuntimeError('Native authored box identity mismatch')
 for name,x in a.items():
  for field in errors:errors[field].append(float(np.max(abs(np.asarray(x[field])-np.asarray(b[name][field])))))
result=dict(scope='Offline USD with anonymous-session primitive authoring; no SimulationApp/PhysX step, no asset save',
    model=model,shape_count=len(actual.shapes),radius_bounds_m=actual.radius_bounds(),max_retained_geometry_error={k:max(v) for k,v in errors.items()},
    asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),author_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
# A held robot may have different USD link poses when another process resumes.
# Model identity must depend on local collider/joint frames, never those poses.
for link in ('gripper_link','moving_jaw_so101_v1_link'):
    UsdGeom.Xformable(stage.GetPrimAtPath(str(root.GetPath())+'/'+link)).AddTranslateOp(opSuffix='modelProbe').Set(Gf.Vec3d(.11,-.07,.25))
result['identity_invariant_to_link_pose']=read_model(stage,str(root.GetPath()))['sha256']==model['sha256']
result['retained_comparison_tolerance']=1e-6
result['comparison_note']='Authored float joint quaternions are normalized by the production model; retained matrices predate normalization. Bound is below the 20 um geometric allowance.'
result['passed']=result['identity_invariant_to_link_pose'] and len(actual.shapes)==8 and all(max(v)<result['retained_comparison_tolerance'] for v in errors.values()) and all(a<=b for a,b in zip(actual.radius_bounds(),POLICY['joint_point_radius_bounds_m']))
import argparse
from storage import atomic_json,confined
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
out=confined(args.output);atomic_json(out,result)
print(json.dumps({k:v for k,v in result.items() if k!='model'}));assert result['passed']
