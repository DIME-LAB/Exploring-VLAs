"""Export a held, verified ROS-collector scene from an explicitly assigned Kit.

Call export_scene(path, wrist_path, top_path) through run_python only during a
lead-granted scene window. This module never connects to, launches or stops Kit.
"""
import hashlib
from pathlib import Path
import shared
from storage import confined,atomic_json


def export_scene(directory,wrist_path=None,top_path=None):
    import collection_runtime
    import omni.graph.core as og
    from pxr import Usd,UsdPhysics
    from isaacsim.core.prims import RigidPrim
    from safety import pose_matrix
    import numpy as np
    from native import usd_matrix,stage_geometry
    r=collection_runtime.runtime()
    if r.busy or r.timeline.is_playing():raise RuntimeError('Scene export requires held collector runtime')
    root=confined(directory);root.mkdir(parents=True,exist_ok=False)
    baseline=r.snapshot();links={}
    for prim in Usd.PrimRange(r.stage.GetPrimAtPath('/World/SO_ARM101')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):links[prim.GetName()]=str(prim.GetPath())
    baseline['links']={}
    for name,path in links.items():
        view=RigidPrim(path,name='inproc_export_'+name,reset_xform_properties=False,prepare_contact_sensors=False)
        view.initialize();position,quaternion=view.get_world_poses()
        baseline['links'][name]=pose_matrix(position[0],quaternion[0]).tolist()
    if not links:raise RuntimeError('No physical robot links')
    cameras={};configuration={}
    for key,graph in (('wrist','ActionGraph_WristCamera'),('top','ActionGraph_WorkspaceCamera')):
        attribute='/Graph/'+graph+'/RGBPublish.inputs:renderProductPath'
        product=str(og.Controller.attribute(attribute).get())
        prim=r.stage.GetPrimAtPath(product)
        targets=prim.GetRelationship('camera').GetTargets()
        if len(targets)!=1 or list(prim.GetAttribute('resolution').Get())!=[640,480]:
            raise RuntimeError('Collector render-product configuration is incomplete: '+key)
        cameras[key]=str(targets[0])
        configuration[key]=dict(source_attribute=attribute,render_product=product,camera=cameras[key])
    for key,requested in (('wrist',wrist_path),('top',top_path)):
        if requested is not None and requested!=cameras[key]:raise RuntimeError('Requested camera differs from live collector configuration')
    for path in cameras.values():
        if not r.stage.GetPrimAtPath(path).IsValid():raise RuntimeError('Missing camera '+path)
    camera_bindings={}
    for key,path in cameras.items():
        parent=r.stage.GetPrimAtPath(path)
        while parent.IsValid() and str(parent.GetPath()) not in links.values():parent=parent.GetParent()
        world=usd_matrix(r.stage.GetPrimAtPath(path))
        if not np.allclose(world[:3,:3].T@world[:3,:3],np.eye(3),atol=1e-6):
            raise RuntimeError('Camera optical frame is not a verified rigid metre transform')
        if parent.IsValid():
            name=next(name for name,p in links.items() if p==str(parent.GetPath()))
            parent_world=usd_matrix(parent)
            if not np.allclose(parent_world[:3,:3].T@parent_world[:3,:3],np.eye(3),atol=1e-6):
                raise RuntimeError('Camera parent link must be unit-scale for measured-pose calibration')
            local=np.linalg.inv(parent_world)@world
            camera_bindings[key]=dict(link=name,matrix=local.tolist())
        else:camera_bindings[key]=dict(link=None,matrix=world.tolist())
    geometry=stage_geometry(r.stage,r.body_paths,links,baseline)
    from articulation_audit import readback
    articulation=readback(r.stage,r.art)
    scene=root/'scene.usdc'
    r.stage.Flatten().Export(str(scene))
    asset_sources={}
    for layer in r.stage.GetUsedLayers():
        p=Path(layer.realPath)
        if p.is_file():asset_sources[p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source=str(p))
    manifest=dict(version=1,scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        baseline=baseline,body_paths=r.body_paths,link_paths=links,camera_paths=cameras,
        geometry=geometry,articulation_audit=articulation,source_assets=asset_sources,camera_bindings=camera_bindings,camera_configuration=configuration,
        cup_geometry_sha256=hashlib.sha256((shared.REPO/'isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json').read_bytes()).hexdigest())
    atomic_json(root/'manifest.json',manifest)
    return dict(scene=str(scene),manifest=str(root/'manifest.json'),links=len(links))
