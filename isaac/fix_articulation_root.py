"""Author the SO-ARM101 world joint as its fixed articulation root, offline.

Run only while the project's Isaac instances are stopped. This changes only the
two ArticulationRootAPI opinions; drives, geometry and joint relationships remain
byte-for-byte equal in a normalized layer serialization.
"""
import argparse
import hashlib
import os
from pathlib import Path


def relocate(stage):
    from pxr import UsdPhysics,Sdf
    root=stage.GetDefaultPrim()
    base=stage.GetPrimAtPath(root.GetPath().AppendChild('base_link'))
    joint=stage.GetPrimAtPath(root.GetPath().AppendChild('root_joint'))
    if not root or not base or not joint.IsA(UsdPhysics.FixedJoint):
        raise ValueError('Expected SO-ARM101 base_link and root_joint')
    fixed=UsdPhysics.Joint(joint)
    if (fixed.GetBody0Rel().GetTargets() or fixed.GetBody1Rel().GetTargets()!=[base.GetPath()] or
        fixed.GetJointEnabledAttr().Get() is not True):
        raise ValueError('Expected an enabled world-to-base fixed joint')
    paths=[base.GetPath(),joint.GetPath()]
    def invariant():
        layer=Sdf.Layer.CreateAnonymous();layer.TransferContent(stage.GetRootLayer())
        other=[]
        for prim in (base,joint):
            other.append([api for api in prim.GetAppliedSchemas() if api!='PhysicsArticulationRootAPI'])
        for path in paths:
            prim=layer.GetPrimAtPath(path)
            if prim.HasInfo('apiSchemas'):prim.ClearInfo('apiSchemas')
        return layer.ExportToString(),other
    before=invariant()
    if base.HasAPI(UsdPhysics.ArticulationRootAPI):base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(joint)
    if invariant()!=before:raise RuntimeError('Root relocation changed another asset opinion')
    roots=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if roots!=[str(joint.GetPath())]:raise RuntimeError('Expected exactly one root API on the world joint')
    return dict(articulation_root_api_paths=roots,unchanged_other_opinions=True)


def main():
    from pxr import Usd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset',type=Path);a=parser.parse_args()
    stage=Usd.Stage.Open(str(a.asset));before=hashlib.sha256(a.asset.read_bytes()).hexdigest()
    result=relocate(stage)
    temporary=a.asset.with_name(a.asset.name+'.fixed.usdc')
    try:
        if not stage.GetRootLayer().Export(str(temporary)):raise RuntimeError('Asset export failed')
        os.replace(temporary,a.asset)
    finally:
        if temporary.exists():temporary.unlink()
    import json
    print(json.dumps(dict(result,before_sha256=before,after_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest())))


if __name__=='__main__':main()
