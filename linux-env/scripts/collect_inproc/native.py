"""Isaac 6.1 adapter. Import only after SimulationApp is running."""
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import shared
from contract import JOINTS
from comparisons import matches
from safety import pose_matrix,collider_in_body
from tick_contract import world_step_errors


def usd_matrix(prim):
    from pxr import UsdGeom,Usd
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def stage_geometry(stage,body_paths,links,baseline):
    from pxr import Usd,UsdGeom,UsdPhysics
    srdf=shared.REPO/'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf'
    aliases=dict(base_link='base',shoulder_link='shoulder',upper_arm_link='upper_arm',
        lower_arm_link='lower_arm',wrist_link='wrist',gripper_link='gripper',
        moving_jaw_so101_v1_link='jaw',camera_mount_link='camera_mount',usb_camera_link='usb_camera')
    canonical=lambda name:aliases.get(name,name)
    names={canonical(Path(p).name):name for name,p in links.items()}
    allowed=[]
    for p in ET.parse(srdf).iter('disable_collisions'):
        if p.attrib['link1'] in names and p.attrib['link2'] in names:
            allowed.append([names[p.attrib['link1']],names[p.attrib['link2']]])
    bodies={**body_paths,**links};reverse={path:name for name,path in bodies.items()}
    boxes=[];unsupported=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        parent=prim
        while parent.IsValid() and str(parent.GetPath()) not in reverse:parent=parent.GetParent()
        if not parent.IsValid():
            # The static infinite ground is checked analytically.
            if 'ground' in str(prim.GetPath()).lower():continue
            unsupported.append(str(prim.GetPath()));continue
        if not prim.IsA(UsdGeom.Cube):unsupported.append(str(prim.GetPath()));continue
        body=reverse[str(parent.GetPath())]
        matrix,half=collider_in_body(usd_matrix(parent),usd_matrix(prim),UsdGeom.Cube(prim).GetSizeAttr().Get())
        boxes.append(dict(name=str(prim.GetPath()),body=body,matrix=matrix.tolist(),
            half=half.tolist()))
    if unsupported:raise RuntimeError('Unsupported enabled colliders: '+repr(unsupported))
    joints=[];children=set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(prim);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
        if not b1 or str(b1[0]) not in links.values():continue
        if not b0:
            # A world-fixed root is represented by its measured world transform.
            if not prim.IsA(UsdPhysics.FixedJoint):raise RuntimeError('Nonfixed world joint')
            continue
        if str(b0[0]) not in reverse:raise RuntimeError('Unmapped native joint parent')
        child,parent=reverse[str(b1[0])],reverse[str(b0[0])]
        def frame(p,q):
            v=q.Get();return pose_matrix(list(p.Get()),[v.GetReal(),*v.GetImaginary()]).tolist()
        revolute=prim.IsA(UsdPhysics.RevoluteJoint)
        if not revolute and not prim.IsA(UsdPhysics.FixedJoint):raise RuntimeError('Unsupported native joint '+str(prim.GetPath()))
        joints.append(dict(name=prim.GetName(),parent=parent,child=child,
            type='revolute' if revolute else 'fixed',
            axis=UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() if revolute else 'Z',
            frame0=frame(j.GetLocalPos0Attr(),j.GetLocalRot0Attr()),
            frame1=frame(j.GetLocalPos1Attr(),j.GetLocalRot1Attr())))
        children.add(child)
    gripper=names.get('gripper')
    if not gripper:raise RuntimeError('Native gripper link is unmapped')
    roots={name:baseline['links'][name] for name in links if name not in children}
    return dict(boxes=boxes,joints=joints,roots=roots,allowed_pairs=allowed,
        gripper_body=gripper,ground_z=0.,ground_allowed=[names['base']],
        finger_pad_shapes=[b['name'] for b in boxes if '/pad_' in b['name']],
        srdf_sha256=hashlib.sha256(srdf.read_bytes()).hexdigest())


class Native:
    def __init__(self,scene,manifest,cameras=True):
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
        import omni.usd
        import carb
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation,RigidPrim
        self.manifest=json.loads(Path(manifest).read_text())
        cup_asset=shared.REPO/'isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json'
        if hashlib.sha256(cup_asset.read_bytes()).hexdigest()!=self.manifest['cup_geometry_sha256']:
            raise RuntimeError('Scene and physical success gate use different cup geometry')
        if hashlib.sha256(Path(scene).read_bytes()).hexdigest()!=self.manifest['scene_sha256']:
            raise RuntimeError('Scene digest differs from baseline manifest')
        omni.usd.get_context().open_stage(str(scene))
        self.stage=omni.usd.get_context().get_stage()
        # Edits affect only this private stage; the source file is never saved.
        for prim in list(self.stage.GetPseudoRoot().GetChildren()):
            if prim.GetName() in ('Graph','Render'):self.stage.RemovePrim(prim.GetPath())
        if not cameras:
            for prim in self.stage.Traverse():
                if prim.IsA(UsdGeom.Camera) and prim.HasAPI('OmniSensorAPI'):prim.RemoveAPI('OmniSensorAPI')
        self.world=World(physics_dt=1/120,rendering_dt=1/30,stage_units_in_meters=1.,backend='numpy',device='cpu')
        physics=self.world.get_physics_context()
        physics.enable_gpu_dynamics(False);physics.set_broadphase_type('MBP')
        self.settings=carb.settings.get_settings()
        self.settings.set('/app/runLoops/main/rateLimitEnabled',False)
        self.settings.set('/app/player/useFixedTimeStepping',True)
        self.world.reset()
        self.world.play()
        self.art=Articulation('/World/SO_ARM101',name='inproc_arm',reset_xform_properties=False)
        self.art.initialize()
        if set(self.art.dof_names)!=set(JOINTS):raise RuntimeError('Native articulation has unexpected degrees of freedom')
        self.indices=[self.art.dof_names.index(n) for n in JOINTS]
        self.bodies={};self.links={}
        for key,paths in (('bodies',self.manifest['body_paths']),('links',self.manifest['link_paths'])):
            dest=self.bodies if key=='bodies' else self.links
            for name,path in paths.items():
                view=RigidPrim(path,name='inproc_'+key+'_'+name,reset_xform_properties=False,prepare_contact_sensors=False)
                view.initialize();dest[name]=view
        self.epoch=0;self.floor=-1;self.last_apply=None
        self.local_corners=self.manifest['baseline']['local_corners']
        self.rgb=None
        if cameras:self.create_cameras()
        self.restore(self.manifest['baseline'])
        scene_prim=self.stage.GetPrimAtPath(physics.prim_path)
        if PhysxSchema.PhysxSceneAPI(scene_prim).GetEnableGPUDynamicsAttr().Get() is not False:
            raise RuntimeError('CPU PhysX readback failed')

    def create_cameras(self):
        from pxr import UsdGeom
        from rgb_capture import RGB
        for key,path in self.manifest['camera_paths'].items():
            if not self.stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):raise RuntimeError('Missing camera '+path)
        self.rgb=RGB(self.manifest['camera_paths'])

    def stamp(self):return round(float(self.world.current_time)*1e9)

    def snapshot(self):
        bodies={}
        for name,view in self.bodies.items():
            p,q=view.get_world_poses()
            bodies[name]=dict(position=p[0].astype(float).tolist(),orientation=q[0].astype(float).tolist(),
                velocity=view.get_velocities()[0].astype(float).tolist())
        links={}
        for name,view in self.links.items():
            p,q=view.get_world_poses();links[name]=pose_matrix(p[0],q[0]).tolist()
        return dict(epoch=self.epoch,sim_ns=self.stamp(),physics_time_s=float(self.world.current_time),
            state=self.art.get_joint_positions()[0,self.indices].astype(float).tolist(),
            velocity=self.art.get_joint_velocities()[0,self.indices].astype(float).tolist(),
            drive_targets=self.art.get_applied_actions().joint_positions[0,self.indices].astype(float).tolist(),
            state_source='isaac_articulation',bodies=bodies,links=links,local_corners=self.local_corners,
            base_world_pose=self.manifest['baseline']['base_world_pose'])

    def apply(self,command,stamp,epoch):
        if epoch!=self.epoch or stamp!=self.stamp():raise RuntimeError('Stale in-process command')
        if self.last_apply==(epoch,stamp):raise RuntimeError('Second command for the same observation')
        values=np.asarray(command,np.float32).reshape(1,6)
        if not np.isfinite(values).all():raise ValueError('Non-finite command')
        self.art.set_joint_position_targets(values,joint_indices=self.indices)
        self.art.set_joint_velocity_targets(np.zeros((1,6),np.float32),joint_indices=self.indices)
        applied=self.art.get_applied_actions().joint_positions[0,self.indices].astype(float).tolist()
        if not matches(applied,list(command),'target_rad'):raise RuntimeError('Drive target readback differs')
        self.last_apply=(epoch,stamp)
        return dict(application_stamp_ns=self.stamp(),applied_command=applied)

    def step(self):
        start=time.perf_counter_ns();old=float(self.world.current_time);index=self.world.current_time_step_index
        for _ in range(4):self.world.step(render=False,update_fabric=True)
        errors=world_step_errors(float(self.world.current_time),old)
        if errors or self.world.current_time_step_index-index!=4:raise RuntimeError('Physical step failed: '+repr(errors))
        return (time.perf_counter_ns()-start)/1e6

    def render(self):
        import omni.timeline
        before=self.snapshot()
        omni.timeline.get_timeline_interface().set_current_time(before['physics_time_s'])
        # Zero delta explicitly prevents capture from introducing a fifth substep.
        images,profile=self.rgb.read(before['physics_time_s'])
        after=self.snapshot()
        if before!=after:raise RuntimeError('Rendering mutated the frozen physical observation')
        return images,profile

    def transforms(self):
        s=self.snapshot()
        transforms={self.manifest['link_paths'][name]:matrix for name,matrix in s['links'].items()}
        transforms.update({self.manifest['body_paths'][name]:pose_matrix(b['position'],b['orientation']).tolist()
            for name,b in s['bodies'].items()})
        cameras={}
        for key,binding in self.manifest['camera_bindings'].items():
            parent=np.asarray(s['links'][binding['link']]) if binding['link'] else np.eye(4)
            cameras[key]=(parent@np.asarray(binding['matrix'])).tolist()
        return transforms,cameras

    def restore(self,baseline):
        self.world.pause()
        self.epoch+=1;self.floor=self.stamp();self.last_apply=None
        if set(baseline['bodies'])!=set(self.bodies):raise RuntimeError('Restore body inventory differs')
        for name,b in baseline['bodies'].items():
            self.bodies[name].set_world_poses(np.asarray([b['position']],np.float32),np.asarray([b['orientation']],np.float32))
            self.bodies[name].set_velocities(np.zeros((1,6),np.float32))
        self.art.set_joint_positions(np.asarray([baseline['state']],np.float32),joint_indices=self.indices)
        self.art.set_joint_velocities(np.zeros((1,6),np.float32),joint_indices=self.indices)
        self.apply(baseline['state'],self.stamp(),self.epoch)
        self.last_apply=None
        s=self.snapshot()
        for name,b in s['bodies'].items():
            requested=baseline['bodies'][name]
            if not matches(b['position'],requested['position'],'float32_readback') or not (
                matches(b['orientation'],requested['orientation'],'float32_readback') or
                matches(b['orientation'],[-v for v in requested['orientation']],'float32_readback')):
                raise RuntimeError('Restore pose readback differs: '+name)
            if not matches(b['velocity'],[0.]*6,'float32_readback'):raise RuntimeError('Restore body velocity differs')
        if not matches(s['state'],baseline['state'],'target_rad') or not matches(s['velocity'],[0.]*6,'float32_readback'):
            raise RuntimeError('Restore arm readback differs')
        boundary=dict(epoch=self.epoch,sim_ns=self.floor,physics_time_s=s['physics_time_s'],drive_targets=s['drive_targets'])
        self.world.play()
        return dict(epoch=self.epoch,invalid_through_ns=self.floor,arm_restored=True,
            snapshot=s,command_baseline=boundary)

    def close(self):
        self.world.pause()
        if self.rgb:self.rgb.close()
