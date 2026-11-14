import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from core import Capture
from playing import Playing,FrameLedger
from tick_contract import STEP_NS,verify_proof
from test_core import collision


class PrimingTests(unittest.TestCase):
    def fixture(self,counts=(0,2,1,1,1,1,1),bad_stamp=False):
        owner=Playing.__new__(Playing)
        owner.frames=FrameLedger();owner.last_apply=None;owner.pending=None;owner.epoch=3
        owner.priming_reports=[];owner.counter=0;owner.copy_count=0
        owner.s=dict(epoch=3,sim_ns=1000000000,physics_time_s=1.,state=[0.]*6,
            velocity=[0.]*6,drive_targets=[0.]*6,bodies={},state_source='isaac_articulation')
        owner.snapshot=lambda:copy.deepcopy(owner.s)
        owner.clock=lambda:{k:owner.s[k] for k in ('epoch','sim_ns','physics_time_s')}
        owner.stamp=lambda:owner.s['sim_ns']
        restored=dict(epoch=3,invalid_through_ns=owner.s['sim_ns'],arm_restored=True,
            snapshot=owner.snapshot(),command_baseline=dict(owner.clock(),drive_targets=[0.]*6))
        def advance():
            before=owner.clock();owner.counter+=1
            stamp=round((1+owner.counter/30)*1e9)
            owner.s.update(sim_ns=stamp,physics_time_s=1+owner.counter/30)
            count=counts[min(owner.counter-1,len(counts)-1)];owner.copy_count+=count
            owner.last_frame=dict(owner.clock(),timeline_ns=stamp,fabric_time_s=stamp/1e9,
                native_physics_steps=4,world_step_count=4,
                render_completions={key:count for key in ('wrist','top')})
            owner.frames.check_advance(owner.last_frame,before)
            owner.image_stamp=before['sim_ns']-(STEP_NS if bad_stamp else 0)
            return owner.last_frame,{},1.
        owner.advance=advance
        def read(counts_before):
            count=list(owner.last_frame['render_completions'].values())[0]
            owner.rgb.calls=count
            return ({key:np.zeros((480,640,3),np.uint8) for key in ('wrist','top')},
                {key:dict(render_frame_number=owner.counter,copy_compute_count=owner.copy_count,
                    stamp_ns=owner.image_stamp,completions=count) for key in ('wrist','top')},0.)
        owner.rgb=SimpleNamespace(read=read,calls=0)
        def apply(values,stamp,epoch):
            self.assertEqual((stamp,epoch),(owner.s['sim_ns'],owner.s['epoch']))
            self.assertNotEqual(owner.last_apply,(epoch,stamp))
            owner.last_apply=(epoch,stamp);owner.s['drive_targets']=list(values)
            return dict(application_stamp_ns=stamp,applied_command=list(values))
        owner.apply=apply
        return owner,restored

    def test_transient_is_discarded_then_first_rows_use_the_shared_contract(self):
        owner,restored=self.fixture()
        boundary=owner.prime(restored)
        self.assertEqual(owner.counter,5);self.assertTrue(boundary['priming']['passed'])
        self.assertIsNone(owner.last_apply);self.assertIsNone(owner.pending)
        self.assertTrue(all(not r['accepted'] for r in boundary['priming']['updates']))
        capture=Capture(owner,collision);capture.boundary(boundary)
        owner.step() # First observation after the verified settled boundary.
        for _ in range(2):
            stamp=owner.s['sim_ns']
            row,checked,_,_,_,_=capture.playing_tick([0.]*6,[0.]*6,[0.]*6,'hold','test',owner.snapshot())
            self.assertEqual(row['sim_ns'],stamp)
            self.assertEqual(row['wrist_stamp_ns'],stamp)
            self.assertFalse(verify_proof(checked['tick_contract']))
        self.assertEqual(len(boundary['priming']['updates']),5)
        # A later reset also primes; cancellation is checked against the last
        # recorded row before its discarded multi-update interval is opened.
        owner.s['epoch']+=1;owner.epoch=owner.s['epoch'];owner.last_apply=None
        restored=dict(epoch=owner.s['epoch'],invalid_through_ns=owner.s['sim_ns'],
            arm_restored=True,snapshot=owner.snapshot(),
            command_baseline=dict(owner.clock(),drive_targets=[0.]*6))
        boundary=owner.prime(restored);capture.boundary(boundary);owner.step()
        row,checked,_,_,_,_=capture.playing_tick([0.]*6,[0.]*6,[0.]*6,'hold','reset',owner.snapshot())
        self.assertEqual(row['epoch'],4)
        self.assertFalse(verify_proof(checked['tick_contract']))

    def test_original_restore_is_checked_before_discarding_the_old_ledger(self):
        owner,restored=self.fixture();capture=Capture(owner,collision)
        capture.contract.previous=dict(epoch=2,sim_ns=restored['invalid_through_ns'],
            pose_stamp_ns=restored['invalid_through_ns'],physics_time_s=1.)
        boundary=owner.prime(restored);capture.boundary(boundary)
        owner,restored=self.fixture();capture=Capture(owner,collision)
        capture.contract.previous=dict(epoch=2,sim_ns=restored['invalid_through_ns']-2*STEP_NS,
            pose_stamp_ns=restored['invalid_through_ns']-2*STEP_NS,physics_time_s=1.-2/30)
        boundary=owner.prime(restored)
        with self.assertRaises(RuntimeError):capture.boundary(boundary)

    def test_wrong_stamps_or_no_frames_exhaust_the_explicit_bound(self):
        for counts,bad_stamp in (((0,),False),((1,),True),((1,1,2),False)):
            owner,restored=self.fixture(counts,bad_stamp)
            with self.assertRaisesRegex(RuntimeError,'did not stabilize'):
                owner.prime(restored,max_updates=6)
            self.assertEqual(owner.counter,6)
            self.assertFalse(owner.priming_reports[-1]['passed'])
            self.assertIsNone(owner.last_apply)

    def test_drive_or_physics_changes_are_terminal_during_priming(self):
        for corruption in ('targets','epoch','steps'):
            owner,restored=self.fixture()
            advance=owner.advance
            def corrupt():
                packet,counts,ms=advance()
                if corruption=='targets':owner.s['drive_targets'][0]=.1
                if corruption=='epoch':owner.s['epoch']+=1
                if corruption=='steps':
                    packet['native_physics_steps']=3
                    owner.frames.check_advance(packet,restored['snapshot'])
                return packet,counts,ms
            owner.advance=corrupt
            with self.assertRaises(RuntimeError):owner.prime(restored)
            self.assertEqual(owner.counter,1)

    def test_priming_never_relaxes_a_later_recorded_gap(self):
        owner,restored=self.fixture((0,2,1,1,1,1,0))
        boundary=owner.prime(restored);capture=Capture(owner,collision)
        capture.boundary(boundary);owner.step()
        with self.assertRaisesRegex(RuntimeError,'exactly one'):
            capture.playing_tick([0.]*6,[0.]*6,[0.]*6,'hold','test',owner.snapshot())
        self.assertIsNone(capture.contract.previous)

    def test_real_joined_v2_trace_stabilizes_after_the_unrecorded_burst(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/joined_v2_startup.json').read_text())
        trace=fixture['trace'];start=fixture['failed_playing_frame']
        owner,restored=self.fixture()
        restored['snapshot'].update(epoch=start['epoch'],sim_ns=start['sim_ns']-STEP_NS,
            physics_time_s=start['physics_time_s']-1/30)
        restored.update(epoch=start['epoch'],invalid_through_ns=restored['snapshot']['sim_ns'],
            command_baseline=dict(restored['snapshot']))
        owner.s=copy.deepcopy(restored['snapshot'])
        rows=[dict(start,cameras=None),*[dict(epoch=start['epoch'],sim_ns=r['after_ns'],
            timeline_ns=r['after_ns'],fabric_time_s=r['after_ns']/1e9,
            physics_time_s=start['physics_time_s']+(i+1)/30,
            native_physics_steps=r['physical_steps'],world_step_count=r['world_steps'],
            render_completions=r['render_completions'],cameras=r['cameras']) for i,r in enumerate(trace)]]
        def advance():
            row=copy.deepcopy(rows[owner.counter]);owner.counter+=1
            owner.frames.check_advance(row,owner.clock());owner.s.update({k:row[k] for k in ('epoch','sim_ns','physics_time_s')})
            owner.last_frame=row
            return row,{},1.
        owner.advance=advance
        owner.rgb.read=lambda counts:({},owner.last_frame['cameras'],0.)
        boundary=owner.prime(restored)
        self.assertEqual(owner.counter,5)
        self.assertEqual(boundary['priming']['consecutive'],3)
        self.assertEqual(boundary['priming']['updates'][-1]['packet']['cameras'],trace[3]['cameras'])
        Capture(owner,collision).boundary(boundary)


if __name__=='__main__':unittest.main()
