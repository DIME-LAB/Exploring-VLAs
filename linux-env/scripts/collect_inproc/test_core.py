import copy
import unittest
import numpy as np
import shared
from core import Capture, command
from replay_contract import packet, verify
from tick_contract import row_errors, verify_proof, STEP_NS
from contract import parameters, timing_audit
from dart import SmoothDart


class Backend:
    """Frozen physical rows with drive readback, shaped like the native runtime."""
    def __init__(self):
        self.s = dict(epoch=1, sim_ns=STEP_NS, physics_time_s=1/30,
            state=[0.]*6, velocity=[0.]*6, drive_targets=[0.]*6,
            bodies={}, state_source='isaac_articulation')
        self.applications=[]
    def snapshot(self): return copy.deepcopy(self.s)
    def clock(self):return {k:self.s[k] for k in ('epoch','sim_ns','physics_time_s')}
    def apply(self, values, stamp, epoch):
        if (stamp, epoch) != (self.s['sim_ns'], self.s['epoch']):
            raise RuntimeError('Stale command')
        self.s['drive_targets'] = list(values)
        self.applications.append((stamp,epoch,list(values)))
        return dict(application_stamp_ns=stamp, applied_command=list(values))


def collision(candidate, snapshot):
    return dict(passed=True, epoch=candidate.epoch, sim_ns=candidate.stamp_ns,
        target=list(candidate.positions), method='test fixture')


class CoreTests(unittest.TestCase):
    def capture(self):
        b = Backend(); c = Capture(b, collision)
        baseline = dict(epoch=1, sim_ns=0, physics_time_s=0., drive_targets=[0.]*6)
        c.boundary(dict(epoch=1, invalid_through_ns=0, arm_restored=True,
            command_baseline=baseline, snapshot=dict(baseline)))
        return b, c
    def test_native_row_passes_existing_contract(self):
        b, c = self.capture()
        row, snap, _ = c.tick([.1]*6, [.1]*6, [.001]*5+[0.], 'move', 'test')
        self.assertEqual(row['action'], [.035]*6)
        self.assertEqual(row['executed_action'], [.101]*5+[.035])
        self.assertFalse(row_errors(row))
        self.assertFalse(verify_proof(snap['tick_contract']))
    def test_clean_label_recomputed_from_perturbed_state(self):
        b, c = self.capture(); b.s['state']=[.07]*6
        row, _, _ = c.tick([.1]*6, [.1]*6, [.001]*5+[0.], 'move', 'test')
        self.assertEqual(row['action'], [.1]*6)
        bad=copy.deepcopy(row);bad['action']=list(bad['executed_action'])
        self.assertIn('Noisy or stale expert label', row_errors(bad))
    def test_original_camera_clock_is_retained(self):
        b,c=self.capture();stamps=dict(wrist=b.s['sim_ns']+167,top=b.s['sim_ns'])
        row,_,_=c.tick([0.]*6,[0.]*6,[0.]*6,'hold','test',camera_stamps=stamps)
        self.assertEqual(row['wrist_stamp_ns'],stamps['wrist'])
        self.assertFalse(row_errors(row))
    def images(self):return {key:np.zeros((480,640,3),np.uint8) for key in ('wrist','top')}
    def test_one_update_join_uses_frozen_state_and_the_real_earlier_ack(self):
        b,c=self.capture();stamp=b.s['sim_ns']
        token=c.prepare([.1]*6,[.1]*6,[.001]*5+[0.],'move','joined')
        self.assertIsNone(c.contract.previous)
        b.s.update(sim_ns=stamp+STEP_NS,physics_time_s=2/30,state=[.02]*6)
        row,snapshot,timing=c.complete(token,self.images(),dict(wrist=stamp-1,top=stamp))
        self.assertEqual(row['state'],[0.]*6);self.assertEqual(row['action'],[.035]*6)
        self.assertEqual(row['executed_action'],[.101]*5+[.035])
        self.assertEqual(row['application_stamp_ns'],stamp)
        self.assertEqual(row['wrist_stamp_ns'],stamp-1)
        self.assertEqual(len(b.applications),1)
        self.assertEqual(timing['image_delay_updates'],1)
        self.assertFalse(row_errors(row));self.assertFalse(verify_proof(snapshot['tick_contract']))
        with self.assertRaises(RuntimeError):c.complete(token,self.images(),dict(wrist=stamp,top=stamp))
    def test_missing_stale_or_cross_epoch_join_never_finalizes(self):
        for mutation in ('late','epoch','stamp','missing','world_frozen'):
            with self.subTest(mutation=mutation):
                b,c=self.capture();stamp=b.s['sim_ns']
                token=c.prepare([0.]*6,[0.]*6,[0.]*6,'hold','joined')
                stamps=dict(wrist=stamp,top=stamp);images=self.images()
                if mutation=='late':b.s['sim_ns']+=2*STEP_NS
                if mutation=='epoch':b.s['epoch']+=1
                if mutation=='stamp':stamps['wrist']+=STEP_NS
                if mutation=='missing':del images['wrist']
                if mutation=='world_frozen':b.s['sim_ns']+=STEP_NS
                with self.assertRaises((RuntimeError,ValueError)):c.complete(token,images,stamps)
                self.assertIsNone(c.contract.previous)
                with self.assertRaises(RuntimeError):c.prepare([0.]*6,[0.]*6,[0.]*6,'hold','joined')
                self.assertEqual(len(b.applications),1)
    def test_playing_driver_advances_once_for_immediate_and_delayed_images(self):
        for delay in (0,1):
            b,c=self.capture();b.calls=[]
            original_apply=b.apply
            def apply(*args):b.calls.append('apply');return original_apply(*args)
            b.apply=apply
            b.image_stamp=b.s['sim_ns'] if delay==0 else None
            b.frame_ready=lambda expected:b.image_stamp==expected['sim_ns']
            def step():
                b.calls.append('step');stamp=b.s['sim_ns']
                b.s.update(sim_ns=stamp+STEP_NS,physics_time_s=b.s['physics_time_s']+1/30,state=[.02]*6)
                b.image_stamp=b.s['sim_ns'] if delay==0 else stamp
                return 1.
            def render(expected):
                b.calls.append('render')
                self.assertEqual(b.image_stamp,expected['sim_ns'])
                stamps=dict(wrist=b.image_stamp,top=b.image_stamp);b.image_stamp=None
                return self.images(),dict(camera_stamps=stamps)
            b.step=step;b.render=render
            row,snapshot,timing,_,_,_=c.playing_tick([.1]*6,[.1]*6,[0.]*6,'move','driver',b.snapshot())
            self.assertEqual(b.calls,['apply','render','step'] if delay==0 else ['apply','step','render'])
            self.assertEqual(len(b.applications),1);self.assertEqual(row['state'],[0.]*6)
            self.assertEqual(timing['image_delay_updates'],delay)
            self.assertFalse(verify_proof(snapshot['tick_contract']))
    def test_skip_and_bad_drive_readback_refuse(self):
        b, c = self.capture(); b.s['sim_ns'] += STEP_NS
        with self.assertRaises(RuntimeError): c.tick([0.]*6,[0.]*6,[0.]*6,'hold','test')
        b, c = self.capture()
        b.apply=lambda *a:dict(application_stamp_ns=b.s['sim_ns'],applied_command=[1.]*6)
        with self.assertRaises(RuntimeError): c.tick([0.]*6,[0.]*6,[0.]*6,'hold','test')
    def test_image_layout_and_gripper_noise_refuse(self):
        b,c=self.capture()
        with self.assertRaises(ValueError):
            c.tick([0.]*6,[0.]*6,[0.]*6,'hold','test',images={'top':np.zeros((480,640,4),np.uint8)})
        with self.assertRaises(ValueError): command(b.s,[0.]*6,[0.]*6,[.1]*6)
    def test_replay_inventory_digest_and_cadence(self):
        b=Backend();m=np.eye(4).tolist()
        a=packet(b.s,{'/arm/link':m},{'wrist':m,'top':m},0)
        verify(a, paths=['/arm/link'])
        b.s['sim_ns']+=STEP_NS
        z=packet(b.s,{'/arm/link':m},{'wrist':m,'top':m},1)
        verify(z,a)
        with self.assertRaises(ValueError):verify(z,a,paths=['/missing'])
        z['payload']['state'][0]=1.
        with self.assertRaises(ValueError):verify(z,a)
    def test_seed_and_dart_identical(self):
        p=parameters(123,'blue',regrasp=True)
        a=SmoothDart(5,p['noise_seed'],p['dart'])
        b=SmoothDart(5,p['noise_seed'],p['dart'])
        self.assertEqual(a.spec,b.spec)
        for i in range(150):self.assertEqual(a.sample(i/30),b.sample(i/30))
        self.assertTrue(.02<=abs(p['regrasp']['signed_offset_m'])<=.04)


if __name__ == '__main__': unittest.main()
