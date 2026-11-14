"""Shared primitive geometry, full-box rest contact, compound MoveIt and CPU gate."""
import ast,copy,math,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from cup_geometry import CUP,PRIMITIVES,WALLS,PROFILE,wall_checks,primitive_world_poses,compound_pose_check
from pose_math import rotate,multiply
from success import contained
from scene_sync import physical_pose_caches,collision_pose_errors
import test_safety


def pose(p,q):return NS(position=NS(**dict(zip(('x','y','z'),p))),orientation=NS(**dict(zip(('w','x','y','z'),q))))
def collision(drop):
    expected=primitive_world_poses(drop)
    return NS(id='cup_drop_3',header=NS(frame_id='base'),meshes=[],mesh_poses=[],
        primitives=[NS(type=1,dimensions=e['dimensions']) for e in expected],
        primitive_poses=[pose(e['position'],e['orientation']) for e in expected])


class PrimitiveCupTests(unittest.TestCase):
    def test_asset_planes_are_the_actual_inner_faces_of_the_twelve_boxes(self):
        self.assertEqual(len(PRIMITIVES),13)
        for wall in WALLS:
            n=wall['interior_plane']['normal'];length=math.sqrt(sum(v*v for v in n));normal=rotate(wall['quaternion_wxyz'],[1,0,0])
            for a,b in zip(n,normal):self.assertAlmostEqual(a/length,b,delta=1e-12)
            inside=[a-b*wall['size_m'][0]/2 for a,b in zip(wall['center_m'],normal)]
            self.assertAlmostEqual(sum(a*b for a,b in zip(n,inside)),wall['interior_plane']['offset_m'],delta=1e-12)
            self.assertEqual(len(wall_checks([inside])),12)

    def test_measured_wall_and_floor_contact_penetration_passes_but_protrusion_does_not(self):
        sample=test_safety.SafetyTests().snapshot(.0012+.0064);name='red_2x3_0'
        plane=WALLS[0]['interior_plane'];n=plane['normal'];length=math.sqrt(sum(v*v for v in n))
        # Set the limiting lowest corner 9 um into wall_00, using its NON-unit n.
        target_x=(plane['offset_m']-n[2]*.0012+9e-6*length)/n[0]
        sample['bodies'][name]['position'][0]=target_x-.0155
        check=contained(sample,name,'red');self.assertTrue(check['passed'],check)
        self.assertAlmostEqual(check['min_wall_clearance_m'],-9e-6,delta=1e-12)
        sample['bodies'][name]['position'][2]-=9e-6
        self.assertTrue(contained(sample,name,'red')['passed'])
        sample['bodies'][name]['position'][0]+=.0001
        self.assertFalse(contained(sample,name,'red')['passed'])
        for kind in ('rim','outside','protruding','below_floor'):
            bad=test_safety.SafetyTests().snapshot(.012)
            if kind=='rim':bad['bodies'][name]['position'][2]=CUP['rim_m']+.0064
            elif kind=='outside':bad['bodies'][name]['position'][0]=.1
            elif kind=='protruding':bad['bodies'][name]['position'][0]=.025
            else:bad['bodies'][name]['position'][2]=.0011+.0064
            with self.subTest(kind=kind):self.assertFalse(contained(bad,name,'red')['passed'])

    def test_compound_pose_uses_cup_base_offset_rotation_and_a_shape_bijection(self):
        base=dict(position=[.04,-.02,.003],orientation=[math.cos(.2),0,0,math.sin(.2)])
        q=[math.cos(.1),0,math.sin(.1),0];position=[.21,.03,.001]
        snap=dict(base_world_pose=base,bodies={'cup_red':dict(position=position,orientation=q,velocity=[0]*6)})
        _,drops=physical_pose_caches(snap,primitive_cups=True);drop=drops['drop_3'];co=collision(drop)
        self.assertTrue(collision_pose_errors([co],snap,primitive_cups=True)['passed'])
        # Floor box centre must be 0.6 mm above cup base, not drop/body centre.
        inv=[base['orientation'][0]]+[-v for v in base['orientation'][1:]]
        relative=rotate(inv,[a-b for a,b in zip(position,base['position'])]);rq=multiply(inv,q)
        expected=[a+b for a,b in zip(relative,rotate(rq,[0,0,.0006]))]
        actual=co.primitive_poses[0].position
        for axis,value in zip(('x','y','z'),expected):self.assertAlmostEqual(getattr(actual,axis),value,delta=1e-12)
        co.primitives.reverse();co.primitive_poses.reverse()
        self.assertTrue(compound_pose_check(co,drop,snap)['passed'])
        co.primitives[0].dimensions[1]+=.0001
        self.assertFalse(compound_pose_check(co,drop,snap)['passed'])
        co=collision(drop);co.primitive_poses[0].position.z+=.04826
        self.assertFalse(compound_pose_check(co,drop,snap)['passed'])
        co=collision(drop);co.primitives[0].dimensions[1]=math.nan
        self.assertFalse(compound_pose_check(co,drop,snap)['passed'])

    def test_cpu_dynamics_must_be_explicitly_verified(self):
        source=Path(__file__).with_name('collection_runtime.py');tree=ast.parse(source.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='physics_mode')
        for value in (False,True,None):
            namespace=dict(PhysxSchema=NS(PhysxSceneAPI=lambda prim:NS(GetEnableGPUDynamicsAttr=lambda:NS(Get=lambda:value))))
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
            stage=NS(GetPrimAtPath=lambda path:NS(IsValid=lambda:True,HasAPI=lambda api:True))
            if value is False:self.assertEqual(namespace['physics_mode'](stage)['dynamics'],'cpu')
            else:
                with self.assertRaises(RuntimeError):namespace['physics_mode'](stage)

    def test_actual_collector_adapter_roundtrips_all_39_cup_boxes_through_strict_scene_client(self):
        import sys
        from types import MethodType
        from unittest.mock import patch
        from test_scene_lifetime import SceneLifetimeTests,Primitive,pose as blank_pose
        c,r,entry,server,markers=SceneLifetimeTests().setup_scene()
        c.scene_reset.primitive_cups=True
        c.node._add_cup_collision_objects=MethodType(type(c).add_cup_collisions,c)
        c.episode_start_physical=SceneLifetimeTests().fixture()
        modules={'geometry_msgs.msg':NS(Pose=blank_pose),'shape_msgs.msg':NS(SolidPrimitive=Primitive)}
        with patch.dict(sys.modules,modules):c.sync_scene()
        self.assertTrue(c.scene_pose_check['passed'],c.scene_pose_check)
        cups=[co for co in server.scene.world.collision_objects if co.id.startswith('cup_')]
        self.assertEqual(len(cups),3);self.assertEqual(sum(len(co.primitives) for co in cups),39)
        self.assertTrue(all(not co.meshes for co in cups));self.assertEqual(markers,[])
        cups[0].primitive_poses[0].position.z+=.001
        self.assertFalse(c.scene_reset.verify(c.episode_start_physical,list(c.node._actual_positions))['passed'])

    def test_fresh_offset_gate_matches_authored_box_attributes_and_rejects_mesh_era_values(self):
        root=Path(__file__).resolve().parents[3]
        source=root/'isaac/exts/soarm101-dt/so_arm101_dt/primitive_colliders.py';tree=ast.parse(source.read_text())
        constant=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTACT_OFFSET_M' for t in n.targets))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collision');authored={}
        api=NS(CreateContactOffsetAttr=lambda:NS(Set=lambda v:authored.update(contact=v)),
               CreateRestOffsetAttr=lambda:NS(Set=lambda v:authored.update(rest=v)))
        env=dict(UsdPhysics=NS(CollisionAPI=NS(Apply=lambda prim:NS(CreateCollisionEnabledAttr=lambda:NS(Set=lambda v:None)))),
            PhysxSchema=NS(PhysxCollisionAPI=NS(Apply=lambda prim:api)),UsdGeom=NS(Imageable=lambda prim:NS(MakeInvisible=lambda:None)))
        exec(compile(ast.Module(body=[constant,fn],type_ignores=[]),str(source),'exec'),env);env['collision'](object())
        self.assertEqual(authored,dict(contact=.0001,rest=0.))
        collector=Path(__file__).with_name('collect.py');parsed=ast.parse(collector.read_text())
        run=next(n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name=='run')
        loop=next(n for n in run.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [v.id for v in n.target.elts]==['name','values'])
        offsets={'red_2x3_0':dict(contact_m=[authored['contact']],rest_m=[authored['rest']]),
                 'cup_red':dict(contact_m=[authored['contact']]*13,rest_m=[authored['rest']]*13)}
        env=dict(offsets=offsets,PROFILE=PROFILE,PRIMITIVES=PRIMITIVES)
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(collector),'exec'),env)
        offsets['red_2x3_0']['contact_m']=[.0013625002466142178]
        with self.assertRaisesRegex(RuntimeError,'Effective shape offsets differ'):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),str(collector),'exec'),env)
