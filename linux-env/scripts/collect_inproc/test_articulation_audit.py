import unittest
from types import SimpleNamespace
from articulation_audit import readback


class ArticulationAuditTests(unittest.TestCase):
    def test_base_mode_comes_from_physx_not_the_six_joint_dofs_or_usd_joint(self):
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/SO_ARM101')
        base=UsdGeom.Xform.Define(stage,'/World/SO_ARM101/base_link').GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(base)
        joint=UsdPhysics.FixedJoint.Define(stage,'/World/SO_ARM101/root_joint')
        joint.CreateBody1Rel().SetTargets([base.GetPath()])
        for fixed in (True,False):
            art=SimpleNamespace(_physics_view=SimpleNamespace(shared_metatype=SimpleNamespace(
                fixed_base=fixed,dof_count=6,link_count=11)))
            result=readback(stage,art)
            self.assertEqual(result['fixed_base'],fixed)
            self.assertEqual(result['joint_dof_count'],6)
            self.assertEqual(result['root_dof_count'],0 if fixed else 6)
            self.assertEqual(result['articulation_root_api_paths'],['/World/SO_ARM101/base_link'])
            self.assertEqual(result['world_fixed_joints'][0]['body1'],[str(base.GetPath())])
        art._physics_view.shared_metatype.fixed_base=None
        with self.assertRaises(RuntimeError):readback(stage,art)


if __name__=='__main__':unittest.main()
