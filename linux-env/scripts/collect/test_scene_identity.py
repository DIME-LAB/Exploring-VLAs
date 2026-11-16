"""Exact scene restore uses the shared native setters, not an alternate reset."""
import copy,unittest
from types import SimpleNamespace as NS
import numpy as np
from scene_identity import BODY_NAMES,physical_state,identity,require_identical
from test_review_regressions import load_class


def snapshot():
    return dict(epoch=1,sim_ns=1000000000,physics_time_s=1.,state=[.125]*6,velocity=[0.]*6,
        drive_targets=[.25]*6,base_world_pose=dict(position=[0.]*3,orientation=[1.,0.,0.,0.]),
        robot_base_world_pose=dict(position=[0.]*3,orientation=[1.,0.,0.,0.]),
        bodies={n:dict(position=[.125,.25,.5],orientation=[1.,0.,0.,0.],velocity=[0.]*6) for n in BODY_NAMES})


class SceneIdentityTests(unittest.TestCase):
    def runtime(self):
        Runtime,scope=load_class('collection_runtime.py','Runtime')
        scope.update(physical_state=physical_state,require_identical=require_identical)
        r=object.__new__(Runtime);data=snapshot();r.busy=False;r.epoch=1;r.indices=list(range(6))
        r.invalid_through_ns=0;r.timeline=NS(is_playing=lambda:False);r.require_scene=lambda:None
        def cancel():r.epoch+=1
        r.cancel=cancel;r.stamp=lambda:1000000000
        r.snapshot=lambda:dict(copy.deepcopy(data),epoch=r.epoch)
        r.bodies={}
        for name in BODY_NAMES:
            def pose(p,q,name=name):data['bodies'][name].update(position=p[0].tolist(),orientation=q[0].tolist())
            def velocity(v,name=name):data['bodies'][name]['velocity']=v[0].tolist()
            r.bodies[name]=NS(set_world_poses=pose,set_velocities=velocity)
        r.art=NS(set_joint_positions=lambda x,**kw:data.update(state=x[0].tolist()),
            set_joint_velocities=lambda x,**kw:data.update(velocity=x[0].tolist()))
        r.apply=lambda x,stamp:data.update(drive_targets=np.asarray(x,dtype=np.float32).tolist())
        return r,data

    def test_three_native_restores_have_identical_physical_bits_and_distinct_epochs(self):
        r,data=self.runtime();expected=snapshot();hashes=[];epochs=[]
        for color in ('red','green','blue'):
            data['bodies']['cup_blue']['position'][0]+=1.;data['state'][1]+=.5
            result=r.restore(expected,arm=True,restore_drives=True,exact_state=True)
            self.assertTrue(result['exact_physical_state']['passed'])
            hashes.append(identity(result['snapshot']));epochs.append(result['epoch'])
        self.assertEqual(hashes,[identity(expected)]*3);self.assertEqual(len(set(epochs)),3)

    def test_one_native_ulp_cup_or_arm_or_velocity_or_drive_edit_refuses(self):
        for field in ('cup','joint','velocity','drive','base'):
            r,data=self.runtime();expected=snapshot();read=r.snapshot
            def changed():
                value=read()
                vector=(value['bodies']['cup_blue']['position'] if field=='cup' else
                    value['robot_base_world_pose']['position'] if field=='base' else
                    value[{'joint':'state','velocity':'velocity','drive':'drive_targets'}[field]])
                vector[0]=float(np.nextafter(np.float32(vector[0]),np.float32(np.inf)))
                return value
            r.snapshot=changed
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Exact physical restore differs'):
                r.restore(expected,arm=True,restore_drives=True,exact_state=True)

    def test_default_eval_restore_keeps_its_existing_tolerance_and_zero_velocity(self):
        r,data=self.runtime();expected=snapshot();expected['velocity']=[.125]*6
        for b in expected['bodies'].values():b['velocity']=[.25]*6
        restored=r.restore(expected,arm=True,restore_drives=True)
        self.assertEqual(restored['snapshot']['velocity'],[0.]*6)
        self.assertTrue(all(b['velocity']==[0.]*6 for b in restored['snapshot']['bodies'].values()))
        self.assertNotIn('exact_physical_state',restored)

    def test_hash_excludes_only_clock_identity_and_keeps_quaternion_sign_and_signed_zero(self):
        a=snapshot();b=copy.deepcopy(a);b.update(epoch=7,sim_ns=9999999999,physics_time_s=10.)
        self.assertEqual(identity(a),identity(b))
        b['bodies']['cup_red']['orientation']=[-1.,0.,0.,0.]
        self.assertNotEqual(identity(a),identity(b))
        b=copy.deepcopy(a);b['velocity'][0]=-0.
        self.assertNotEqual(identity(a),identity(b))

    def test_missing_body_or_nonfinite_state_cannot_claim_an_identity(self):
        for field in ('body','nan'):
            a=snapshot()
            if field=='body':a['bodies'].pop('cup_green')
            else:a['state'][0]=float('nan')
            with self.subTest(field=field),self.assertRaises(ValueError):identity(a)


if __name__=='__main__':unittest.main()
