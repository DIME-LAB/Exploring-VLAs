"""Read PhysX base mode and USD root placement without changing the stage."""


def readback(stage,articulation,robot_path='/World/SO_ARM101'):
    from pxr import Usd,UsdPhysics
    # Installed omni.physics.tensors/api.py documents fixed_base on the native
    # metatype; six actuated DOFs alone cannot distinguish a floating base.
    meta=articulation._physics_view.shared_metatype
    fixed=meta.fixed_base
    if not isinstance(fixed,bool):raise RuntimeError('PhysX fixed-base flag is unavailable')
    roots=[];world_joints=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.FixedJoint):
            joint=UsdPhysics.Joint(prim)
            if not joint.GetBody0Rel().GetTargets():
                world_joints.append(dict(path=str(prim.GetPath()),enabled=joint.GetJointEnabledAttr().Get(),
                    body1=[str(path) for path in joint.GetBody1Rel().GetTargets()]))
    return dict(fixed_base=fixed,source='PhysX ArticulationView.shared_metatype.fixed_base',
        joint_dof_count=int(meta.dof_count),link_count=int(meta.link_count),
        root_dof_count=0 if fixed else 6,root_dof_count_source='derived from native fixed_base flag',
        articulation_root_api_paths=roots,world_fixed_joints=world_joints)
