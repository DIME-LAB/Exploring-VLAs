import importlib.util
import unittest
import shared


class FixedRootAssetTests(unittest.TestCase):
    def test_production_asset_has_one_root_on_the_world_fixed_joint(self):
        from pxr import Usd,UsdPhysics
        stage=Usd.Stage.Open(str(shared.REPO/'isaac/exts/soarm101-dt/assets/SO-ARM101-USD.usd'))
        root=stage.GetDefaultPrim()
        paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        self.assertEqual(paths,[str(root.GetPath().AppendChild('root_joint'))])
        names=[p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        from contract import JOINTS
        self.assertEqual(set(names),set(JOINTS))

    def test_relocation_is_idempotent_and_preserves_all_other_asset_opinions(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        path=shared.REPO/'isaac/fix_articulation_root.py'
        spec=importlib.util.spec_from_file_location('fix_articulation_root',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        stage=Usd.Stage.CreateInMemory();root=UsdGeom.Xform.Define(stage,'/robot').GetPrim()
        stage.SetDefaultPrim(root)
        base=UsdGeom.Xform.Define(stage,'/robot/base_link').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base);UsdPhysics.ArticulationRootAPI.Apply(base)
        UsdPhysics.MassAPI.Apply(base).CreateMassAttr(.147)
        joint=UsdPhysics.FixedJoint.Define(stage,'/robot/root_joint')
        joint.CreateBody1Rel().SetTargets([base.GetPath()])
        module.relocate(stage);serialized=stage.GetRootLayer().ExportToString()
        module.relocate(stage)
        self.assertEqual(serialized,stage.GetRootLayer().ExportToString())
        self.assertAlmostEqual(UsdPhysics.MassAPI(base).GetMassAttr().Get(),.147,places=7)


if __name__=='__main__':unittest.main()
