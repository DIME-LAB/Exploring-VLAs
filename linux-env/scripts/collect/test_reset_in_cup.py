"""Full collector reset keeps a real placed block; GUI refresh stays pinned."""
import json,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from unittest.mock import patch
import test_scene_lifetime as lifetime
from test_scene_lifetime import Primitive,pose,JOINTS

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('reset-after-green-drop.json').read_text())


class FullResetTests(unittest.TestCase):
    def test_retained_post_drop_scene_rebuild_includes_all_six_and_strict_policy(self):
        c,r,entry,server,markers=lifetime.SceneLifetimeTests().setup_scene();snapshot=FIXTURE['snapshot'];c.episode_start_physical=snapshot
        c.scene_reset.primitive_cups=True;c.node._add_cup_collision_objects=MethodType(type(c).add_cup_collisions,c)
        server.scene.robot_state.joint_state.position=list(snapshot['state'])
        original=c.scene_reset.rebuild_legos
        # The old sync entrance routes to the unmodified GUI in-cup filter.
        c.scene_reset.rebuild_legos=lambda snap:c.node._add_lego_collision_objects(legos_dict=dict(c.node.objects_data))
        scope=c.sync_scene.__globals__;scope['time']=NS(monotonic=iter((0.,20.)).__next__)
        modules={'geometry_msgs.msg':NS(Pose=pose),'shape_msgs.msg':NS(SolidPrimitive=Primitive)}
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(TimeoutError,'missing collision object'):c.sync_scene()
            self.assertNotIn('lego_green_2x3_0',{o.id for o in server.scene.world.collision_objects})
            c.scene_reset.rebuild_legos=original;scope['time']=NS(monotonic=lambda:0.)
            c.sync_scene()
        self.assertTrue(c.scene_pose_check['passed']);self.assertTrue(c.scene_pose_check['collision_policy'])
        self.assertEqual(len([o for o in server.scene.world.collision_objects if o.id.startswith('lego_')]),6)
        self.assertIn('lego_green_2x3_0',{o.id for o in server.scene.world.collision_objects});self.assertEqual(markers,[])
        block=next(o for o in server.scene.world.collision_objects if o.id=='lego_green_2x3_0');block.primitives[0].dimensions[0]+=.001
        self.assertFalse(c.scene_reset.verify(snapshot,JOINTS)['passed'])
