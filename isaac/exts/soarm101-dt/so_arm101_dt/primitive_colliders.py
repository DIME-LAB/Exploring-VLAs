"""Analytic collision shapes for the SO-ARM101 sorting scene (metres)."""
import json
from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema

CONTACT_OFFSET_M = 0.0001


def collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
    px.CreateRestOffsetAttr().Set(0.0)
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def box(stage, path, low, high, orient=None):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(shape)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[(a+b)/2 for a,b in zip(low,high)]))
    if orient is not None:
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
    xf.AddScaleOp().Set(Gf.Vec3d(*[b-a for a,b in zip(low,high)]))
    return collision(shape.GetPrim())


def disable_mesh_collisions(root):
    """Leave render geometry and rigid-body/inertial properties untouched."""
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)


def lego(stage, body):
    points = UsdGeom.Mesh(body).GetPointsAttr().Get()
    bounds = Gf.Range3d()
    for p in points:
        bounds.UnionWith(Gf.Vec3d(p))
    disable_mesh_collisions(body)
    box(stage, body.GetPath().AppendChild('box_collider'), bounds.GetMin(), bounds.GetMax())


def cup(stage, root):
    spec=json.loads((Path(__file__).resolve().parent.parent/'assets/sorting_collision_geometry.json').read_text())['cup']
    disable_mesh_collisions(root)
    path = str(root.GetPath()) + '/primitive_colliders'
    UsdGeom.Xform.Define(stage, path)
    # Keep cup mass independent of collision-volume approximation.
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr().Set(spec['mass_kg'])
    for shape in spec['primitives']:
        center,size,q=shape['center_m'],shape['size_m'],shape['quaternion_wxyz']
        low=[p-s/2 for p,s in zip(center,size)]
        high=[p+s/2 for p,s in zip(center,size)]
        box(stage,path+'/'+shape['name'],low,high,Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))


def robot(stage, root):
    """Two to four boxes per link; split the stepped finger contact faces."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy','guide'])
    for link in root.GetChildren():
        old = stage.GetPrimAtPath(link.GetPath().AppendChild('collisions'))
        if not old.IsValid():
            continue
        fitted=[]
        for p in Usd.PrimRange(old):
            if p.IsA(UsdGeom.Mesh):
                bound=cache.ComputeRelativeBound(p,link).ComputeAlignedRange()
                fitted.append((p.GetParent().GetName(),list(bound.GetMin()),list(bound.GetMax())))
        disable_mesh_collisions(old)
        parent=str(old.GetPath())+'/primitives'
        UsdGeom.Xform.Define(stage,parent)
        if link.GetName()=='base_link':
            fitted=[
                ('foot',[-.0392,-.05547,0.],[.04781,.05547,.0288]),
                ('housing',[-.04781,-.0257,.0288],[.03621,.02531,.07201]),
            ]
        elif link.GetName()=='gripper_link':
            fitted=[
                ('frame',[-.0352,-.024,-.070],[.0304,.024,.001]),
                ('pad_tip',[-.015,-.003,-.1034],[-.0079,.003,-.094425]),
                ('pad_middle',[-.020,-.0057,-.094425],[-.0099,.0057,-.084425]),
                ('pad_upper',[-.026,-.007,-.084425],[-.0116,.007,-.070]),
            ]
        elif link.GetName()=='moving_jaw_so101_v1_link':
            fitted=[
                ('hinge',[-.0123,-.049,-.0051],[.01,.01,.0429]),
                ('pad_tip',[-.0123,-.081,.0159],[-.006,-.072,.0219]),
                ('pad_middle',[-.0103,-.072,.0132],[.002,-.062,.0246]),
                ('pad_upper',[-.0083,-.062,.0119],[.005,-.049,.0259]),
            ]
        for name,low,high in fitted:
            box(stage,parent+'/'+name,low,high)
