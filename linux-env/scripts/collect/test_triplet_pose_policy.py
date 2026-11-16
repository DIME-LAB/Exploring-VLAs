"""Approved pose bounds, retained native round-trip and member divergence refusals."""
import copy,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
from scene_identity import pose_agreement,require_pose_agreement,identity
from target_triplets import validate_run
import test_scene_identity
import test_target_triplets


class TripletPosePolicyTests(unittest.TestCase):
    def test_retained_native_roundtrip_is_admitted_with_exact_bits_still_reported(self):
        data=json.loads((Path(__file__).parent/'fixtures/native-pose-roundtrip-6ba20ab.json').read_text())
        proof=require_pose_agreement(data['actual'],data['expected'])
        self.assertFalse(proof['exact_bits']['passed']);self.assertEqual(len(proof['exact_bits']['differences']),15)
        self.assertLess(max(b['position_error_m'] for b in proof['per_body'].values()),3.36e-8)
        self.assertLess(max(b['orientation_geodesic_deg'] for b in proof['per_body'].values()),5.09e-6)
    def test_position_euclidean_and_geodesic_bounds_refuse_just_over(self):
        base=test_scene_identity.snapshot();base['bodies']['cup_red'].update(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
        for fraction,passed in ((.999,True),(1.001,False)):
            for kind in ('position','orientation'):
                actual=copy.deepcopy(base)
                if kind=='position':actual['bodies']['cup_red']['position'][0]=fraction*1e-6
                else:
                    half=math.radians(fraction*.001)/2;actual['bodies']['cup_red']['orientation']=[math.cos(half),math.sin(half),0.,0.]
                with self.subTest(kind=kind,fraction=fraction):self.assertEqual(pose_agreement(actual,base)['passed'],passed)
        actual=copy.deepcopy(base);actual['bodies']['cup_red']['position']=[.8e-6,.8e-6,0.]
        self.assertFalse(pose_agreement(actual,base)['passed'])
    def test_joints_drives_velocities_and_bases_remain_exact(self):
        for kind in ('state','drive_targets','velocity','body_velocity','base'):
            base=test_scene_identity.snapshot();actual=copy.deepcopy(base)
            values=(actual['bodies']['cup_red']['velocity'] if kind=='body_velocity' else
                actual['robot_base_world_pose']['position'] if kind=='base' else actual[kind])
            values[0]=float(np.nextafter(np.float32(values[0]),np.float32(1.)))
            with self.subTest(kind=kind):
                self.assertFalse(pose_agreement(actual,base)['passed'])
                self.assertTrue(pose_agreement(actual,base)['strict_differences'])
    def test_actual_runtime_reads_twice_without_second_write_and_default_stays_single_read(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/native-pose-roundtrip-6ba20ab.json').read_text())
        r,_=test_scene_identity.SceneIdentityTests().runtime();reads=[];writes=[]
        actual=dict(fixture['actual'],epoch=1,sim_ns=10**9,physics_time_s=1.)
        def read():reads.append(True);return dict(copy.deepcopy(actual),epoch=r.epoch)
        r.snapshot=read;r.apply=lambda *a:writes.append('drive')
        result=r.restore(fixture['expected'],arm=True,restore_drives=True,triplet_state=True)
        self.assertEqual((len(reads),writes),(2,['drive']))
        proof=result['triplet_physical_state'];self.assertFalse(proof['against_target']['exact_bits']['passed'])
        self.assertTrue(proof['reread_exact_bits']['passed']);self.assertTrue(proof['native_clocks_unchanged'])
        r,_=test_scene_identity.SceneIdentityTests().runtime();original=r.snapshot;reads=[]
        def read_default():reads.append(True);return original()
        r.snapshot=read_default;r.restore(test_scene_identity.snapshot(),arm=True)
        self.assertEqual(len(reads),1)
    def test_actual_runtime_refuses_over_bound_or_changing_clock_on_repeat_read(self):
        for kind in ('position','clock'):
            r,_=test_scene_identity.SceneIdentityTests().runtime();original=r.snapshot;reads=[]
            def read():
                value=original();reads.append(True)
                if kind=='position':value['bodies']['cup_red']['position'][0]+=2e-6
                elif len(reads)>1:value['sim_ns']+=1
                return value
            r.snapshot=read
            with self.subTest(kind=kind),self.assertRaises((ValueError,RuntimeError)):
                r.restore(test_scene_identity.snapshot(),arm=True,restore_drives=True,triplet_state=True)
    def test_runtime_uses_geodesic_limit_not_legacy_quaternion_component_limit(self):
        for fraction,admitted in ((.999,True),(1.001,False)):
            r,_=test_scene_identity.SceneIdentityTests().runtime();base=test_scene_identity.snapshot();base['bodies']['cup_red']['orientation']=[1.,0.,0.,0.]
            original=r.snapshot;half=math.radians(fraction*.001)/2
            def read():
                value=original();value['bodies']['cup_red']['orientation']=[math.cos(half),math.sin(half),0.,0.];return value
            r.snapshot=read
            if admitted:self.assertTrue(r.restore(base,arm=True,restore_drives=True,triplet_state=True)['triplet_physical_state']['against_target']['passed'])
            else:
                with self.assertRaisesRegex(ValueError,'physical agreement'):r.restore(base,arm=True,restore_drives=True,triplet_state=True)
    def test_positive_group_preserves_individual_hashes_and_measured_deviations(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);h.member_offset={1:.2e-6,2:-.2e-6};receipts=h.complete(first_offset_ticks=1)
            group=validate_run(h.root,base=Path(d))
            self.assertTrue(group['complete']);self.assertEqual(len(set(group['first_recorded_physical_sha256'])),3)
            self.assertFalse(receipts[1]['counterfactual']['restore_proof']['exact_bits']['passed'])
            self.assertEqual(len(receipts[2]['counterfactual']['first_recorded_prior']),2)
            self.assertEqual(len(receipts[2]['counterfactual']['restore_prior']),8)
    def test_pairwise_restore_divergence_refuses_even_when_each_member_matches_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);h.member_offset={1:.75e-6,2:-.75e-6}
            with self.assertRaisesRegex(ValueError,'physical agreement'):h.complete()
            self.assertIsNone(h.collector.writer)
    def test_pairwise_bootstrap_and_first_frame_divergence_refuse(self):
        for phase in ('bootstrap','first'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as d:
                h=test_target_triplets.TripletTests().fixture(d)
                def drift(data,index=None):
                    index=h.member_index if index is None else index
                    delta={0:0.,1:.75e-6,2:-.75e-6}[index]
                    data['bodies']['cup_red']['position'][0]=float(np.float32(.125+delta))
                if phase=='bootstrap':h.after_bootstrap=drift
                else:h.before_first_record=drift
                with self.assertRaisesRegex(ValueError,'physical agreement'):h.complete(first_offset_ticks=1)
    def test_copied_frame_fields_do_not_get_the_independent_pose_tolerance(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);h.complete()
            p=h.root/'attempt-0000/frames.jsonl';row=json.loads(p.read_text());row['physical_bodies']['cup_red']['position'][0]+=.25e-6
            p.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'native tick proof'):validate_run(h.root,base=Path(d),sealed=False)

    def test_common_anchor_drift_is_measured_and_bounded_at_both_observation_entrances(self):
        for phase in ('bootstrap','first'):
            for kind in ('position','orientation'):
                for value,admitted in ((2e-6,True),(.000499,True),(.000501,False)) if kind=='position' else ((.002,True),(.499,True),(.501,False)):
                    with self.subTest(phase=phase,kind=kind,value=value),tempfile.TemporaryDirectory() as d:
                        h=test_target_triplets.TripletTests().fixture(d)
                        def drift(data,index=None):
                            body=data['bodies']['cup_red']
                            if kind=='position':body['position'][0]=float(np.float32(.125+value))
                            else:
                                half=math.radians(value)/2;body['orientation']=[math.cos(half),math.sin(half),0.,0.]
                        if phase=='bootstrap':h.after_bootstrap=drift
                        else:h.before_first_record=drift
                        if not admitted:
                            with self.assertRaisesRegex(ValueError,'observation anchor drift exceeded'):h.complete(first_offset_ticks=1)
                            self.assertIsNone(h.collector.writer)
                            self.assertFalse((h.root/'attempt-0000/first-recorded-native.json').exists())
                        else:
                            receipts=h.complete(first_offset_ticks=1);group=validate_run(h.root,base=Path(d))
                            key='bootstrap' if phase=='bootstrap' else 'first_recorded'
                            for member in group['member_deviations']:
                                body=member['anchor_drift'][key]['per_body']['cup_red']
                                measured=body['position_error_m'] if kind=='position' else body['orientation_geodesic_deg']
                                expected=(float(np.float32(.125+value))-.125 if kind=='position' else
                                    math.degrees(2*math.atan2(float(np.float32(math.sin(math.radians(value)/2))),
                                                            float(np.float32(math.cos(math.radians(value)/2))))))
                                self.assertAlmostEqual(measured,expected,delta=1e-12)
                                self.assertTrue(member['anchor_drift'][key]['passed'])
                            self.assertTrue(group['complete']);self.assertEqual(len(receipts),3)

    def test_persisted_anchor_drift_evidence_is_reconstructed_before_seal(self):
        for phase in ('bootstrap','first_recorded'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as d:
                h=test_target_triplets.TripletTests().fixture(d);h.complete(first_offset_ticks=1)
                path=h.root/'attempt-0000/receipt.json';receipt=json.loads(path.read_text())
                receipt['counterfactual'][phase+'_anchor_drift']['per_body']['cup_red']['position_error_m']=.0001
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError,'anchor drift evidence differs'):validate_run(h.root,base=Path(d),sealed=False)


if __name__=='__main__':unittest.main()
