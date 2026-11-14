import copy
import math
import unittest
import numpy as np
from safety import Box,overlaps,Geometry,rotation,collider_in_body
from expert import schedule


class SafetyTests(unittest.TestCase):
    def test_millimetre_body_scale_is_retained_in_metre_colliders(self):
        parent=np.diag([.001,.001,.001,1.]);parent[:3,3]=[.2,.1,.01]
        child=np.eye(4);child[:3,:3]*=[16.,24.,10.];child[:3,3]=[0.,0.,5.]
        matrix,half=collider_in_body(parent,parent@child,1.)
        np.testing.assert_allclose(half,[.008,.012,.005])
        np.testing.assert_allclose(matrix[:3,3],[0.,0.,.005],atol=1e-12)
        np.testing.assert_allclose(matrix[:3,:3],np.eye(3))
    def test_rotated_sat_and_near_miss(self):
        a=Box('a','arm',np.eye(4),np.array([1.,.1,.1]))
        m=rotation('Z',math.pi/2);m[:3,3]=[0.,.8,0.]
        b=Box('b','cup',m,np.array([.1,.1,.1]))
        self.assertFalse(overlaps(a,b))
        b.matrix[:3,3]=[.3,0,0]
        self.assertTrue(overlaps(a,b))
    def geometry(self):
        root=np.eye(4);root[2,3]=1
        spec=dict(boxes=[dict(name='arm/box',body='arm',matrix=np.eye(4).tolist(),half=[.1]*3),
            dict(name='object/box',body='object',matrix=np.eye(4).tolist(),half=[.1]*3)],
            joints=[dict(name='shoulder_pan',parent='base',child='arm',type='revolute',
                axis='Z',frame0=np.eye(4).tolist(),frame1=np.eye(4).tolist())],
            roots=dict(base=root.tolist()),allowed_pairs=[],ground_z=0.,ground_allowed=['base'],
            gripper_body='arm',finger_pad_shapes=[])
        g=Geometry(spec);q=[0.]*6
        snap=dict(state=q,links={k:v.tolist() for k,v in g.fk(q).items()},
            bodies={'object':dict(position=[1.,0,1],orientation=[1.,0,0,0],velocity=[0.]*6)})
        g.update(snap)
        return g,snap
    def test_fk_and_dynamic_obstacles(self):
        g,s=self.geometry();self.assertTrue(g.check([0.]*6)[0])
        s['bodies']['object']['position']=[0,0,1]
        g.update(s);self.assertFalse(g.check([0.]*6)[0])
        s['links']['arm'][0][3]=.001
        with self.assertRaises(RuntimeError):g.update(s)
    def test_measured_root_moves_fk_and_sweeps_in_the_world_frame(self):
        g,s=self.geometry();delta=rotation('Z',.0063);delta[0,3]=.000107
        s['links']={name:(delta@np.asarray(matrix)).tolist() for name,matrix in s['links'].items()}
        g.update(s)
        np.testing.assert_allclose(g.fk(s['state'])['arm'],s['links']['arm'],atol=1e-12)
        self.assertAlmostEqual(g.root_motion['base']['rotation_rad'],.0063)
        # Moving the whole robot near a stationary world obstacle changes the
        # collision answer. Child-only FK errors still fault at the same bound.
        for matrix in s['links'].values():matrix[0][3]+=.999893
        g.update(s);self.assertFalse(g.check(s['state'])[0])
        s['links']['arm'][1][3]+=.001
        with self.assertRaises(RuntimeError):g.update(s)
    def test_measured_root_must_be_a_finite_rigid_transform(self):
        for mutation in ('nan','scale','reflection','homogeneous'):
            g,s=self.geometry()
            if mutation=='nan':s['links']['base'][0][3]=float('nan')
            if mutation=='scale':s['links']['base'][0][0]=1.01
            if mutation=='reflection':s['links']['base'][0][0]=-1
            if mutation=='homogeneous':s['links']['base'][3][3]=0
            with self.assertRaises(RuntimeError):g.update(s)
    def test_only_explicit_pad_contact_allowed(self):
        g,s=self.geometry();s['bodies']['object']['position']=[0,0,1]
        g.selected='object';g.contact=True;g.update(s)
        self.assertFalse(g.check([0.]*6)[0])
        g.spec['finger_pad_shapes']=['arm/box']
        self.assertTrue(g.check([0.]*6)[0])
    def test_schedule_preserves_original_linear_time_law(self):
        s=schedule([0.]*5,[1.]*5,3.)
        self.assertEqual(s.sample(1.5),[.5]*5)
        self.assertEqual(s.sample(3),[1.]*5)


if __name__=='__main__':unittest.main()
