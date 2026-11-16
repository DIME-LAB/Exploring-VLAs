"""Native source bits govern across Python versions; diagnostics are not commands."""
import io,json,math,queue,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from scene_identity import identity,require_pose_agreement
from triplet_boundary import restore_admission
from target_triplets import validate_run
import test_target_triplets
from test_review_regressions import load_class
from tick_contract import STEP_NS


class InterpreterProofTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).parent/'fixtures/native-pose-proof-python312.json').read_text())
    def test_actual_restore_admission_replays_python312_native_proof(self):
        data=self.fixture();self.assertIn('3.12.15',data['producer_python'])
        proof=restore_admission(data['snapshot'],data['check'],data['anchor'])
        self.assertTrue(proof['against_target']['passed'])
        self.assertEqual(proof['against_target']['exact_bits']['actual_sha256'],identity(data['snapshot']))
    def test_derived_diagnostic_rounding_is_not_a_cross_interpreter_gate(self):
        data=self.fixture()
        for key in ('against_target','reread_against_target','reread_agreement'):
            body=data['check']['triplet_physical_state'][key]['per_body']['cup_red']
            body['orientation_geodesic_deg']=math.nextafter(body['orientation_geodesic_deg'],math.inf)
        self.assertTrue(restore_admission(data['snapshot'],data['check'],data['anchor'])['against_target']['passed'])
    def test_changed_source_bits_refuse_with_structured_actual_expected(self):
        data=self.fixture();data['snapshot']['bodies']['cup_red']['position'][0]+=5e-8
        require_pose_agreement(data['snapshot'],data['anchor'])  # Still inside the physical pose limit.
        with self.assertRaisesRegex(ValueError,'Restore pose/reread evidence differs') as raised:
            restore_admission(data['snapshot'],data['check'],data['anchor'])
        evidence=json.loads(str(raised.exception).split(': ',1)[1])
        self.assertTrue(evidence['differences']);self.assertIn('actual',evidence);self.assertIn('expected',evidence)
        self.assertTrue(any('actual_sha256' in d['field'] for d in evidence['differences']))
    def test_locally_recomputed_bound_cannot_be_replaced_by_producer_verdict(self):
        data=self.fixture();data['snapshot']['bodies']['cup_red']['position'][0]+=.00001
        with self.assertRaisesRegex(ValueError,'physical agreement failed'):
            restore_admission(data['snapshot'],data['check'],data['anchor'])
    def test_changed_policy_and_nonfinite_diagnostics_refuse(self):
        for field in ('policy','nan'):
            data=self.fixture();proof=data['check']['triplet_physical_state']['against_target']
            if field=='policy':proof['policy']['position_m']=1.
            else:proof['per_body']['cup_red']['orientation_geodesic_deg']=float('nan')
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Restore pose/reread evidence differs'):
                restore_admission(data['snapshot'],data['check'],data['anchor'])
    def test_stored_producer_false_or_nonbool_refuses_a_locally_valid_pose(self):
        for verdict in (False,0,1,None,'true'):
            data=self.fixture();data['check']['triplet_physical_state']['against_target']['passed']=verdict
            require_pose_agreement(data['snapshot'],data['anchor'])
            with self.subTest(verdict=verdict),self.assertRaisesRegex(ValueError,'producer admission must be bool True'):
                restore_admission(data['snapshot'],data['check'],data['anchor'])
    def test_group_recomputes_nested_pose_diagnostics_and_keeps_byte_seals(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);h.complete(first_offset_ticks=1)
            p=h.root/'attempt-0001/receipt.json';receipt=json.loads(p.read_text())
            def diagnostic(value):
                if isinstance(value,dict):
                    if {'policy','per_body','exact_bits'}.issubset(value):
                        for body in value['per_body'].values():
                            body['orientation_geodesic_deg']=math.nextafter(body['orientation_geodesic_deg'],math.inf)
                    else:
                        for child in value.values():diagnostic(child)
                elif isinstance(value,list):
                    for child in value:diagnostic(child)
            diagnostic(receipt['counterfactual']);p.write_text(json.dumps(receipt))
            self.assertTrue(validate_run(h.root,base=Path(d),sealed=False)['complete'])
            with self.assertRaisesRegex(ValueError,'certificate differs|inputs/configuration changed'):
                validate_run(h.root,base=Path(d))
    def test_visible_rule_diagnostics_rounding_is_allowed_but_target_and_policy_stay_exact(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);h.complete(first_offset_ticks=1)
            p=h.root/'attempt-0001/receipt.json';receipt=json.loads(p.read_text())
            for selection in (receipt['params']['target_selection'],*receipt['counterfactual']['observed_target_rule']['selections'].values()):
                selection['margin_m']=math.nextafter(selection['margin_m'],math.inf)
                selection['candidates'][0]['distance_xy_m']=math.nextafter(selection['candidates'][0]['distance_xy_m'],math.inf)
            p.write_text(json.dumps(receipt));self.assertTrue(validate_run(h.root,base=Path(d),sealed=False)['complete'])
            receipt['params']['target_selection']['object']='red_2x3_1';p.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'task/target/common parameters differ'):
                validate_run(h.root,base=Path(d),sealed=False)
    def test_failed_admission_retains_returned_reread_and_observation_drift(self):
        for phase in ('priming_restore','bootstrap','first_recorded'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as d:
                h=test_target_triplets.TripletTests().fixture(d)
                if phase=='priming_restore':
                    restore=h.collector.restore_physical
                    def refused(code):
                        result=restore(code)
                        if 'triplet_state=True' in code:result['triplet_physical_state']['against_target']['passed']=False
                        return result
                    h.collector.restore_physical=refused
                else:
                    def drift(data,index=None):data['bodies']['cup_red']['position'][0]+=.001
                    if phase=='bootstrap':h.after_bootstrap=drift
                    else:h.before_first_record=drift
                with self.assertRaises(ValueError):h.complete(first_offset_ticks=1)
                path=h.root/'attempt-0000';evidence=json.loads((h.root/'triplet-boundaries/attempt-0000.json').read_text())
                self.assertEqual(evidence['episode_id'],'episode-0');proof=evidence['phases'][phase]
                if phase=='priming_restore':
                    reread=proof['restore']['triplet_physical_state']
                    self.assertIn('reread_snapshot',reread);self.assertTrue(reread['reread_exact_bits']['passed'])
                else:
                    self.assertFalse(proof['anchor_drift']['passed'])
                    self.assertGreater(proof['anchor_drift']['per_body']['cup_red']['position_error_m'],.0005)
                self.assertFalse((path/'first-recorded-native.json').exists());self.assertIsNone(h.collector.writer)
    def test_real_writer_creation_and_first_row_after_triplet_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d);receipt=h.prepare(0);path=h.root/'attempt-0000'
            self.assertFalse(path.exists())
            self.assertTrue((h.root/'triplet-boundaries/attempt-0000.json').exists())
            Writer,scope=load_class('collect.py','EpisodeWriter')
            def encoder(*args,**kwargs):
                def wait(timeout):kwargs['stderr'].close();return 0
                return NS(stdin=io.BytesIO(),wait=wait)
            scope.update(queue=queue,subprocess=NS(Popen=encoder,PIPE=subprocess.PIPE))
            writer=Writer(path,30);h.collector.writer=writer
            try:
                with self.assertRaises(FileExistsError):Writer(path,30)
                h.data['sim_ns']+=STEP_NS;h.data['physics_time_s']+=1/30
                native=h.checked_snapshot();row=dict(native['tick_contract']['row'])
                row['wall_profile']=dict(profile_id='cpu-writer-test',epoch=row['epoch'],sim_ns=row['sim_ns'])
                h.collector.triplet_session.record_start(native,row,path)
                writer.add(row,{})
            finally:writer.close();h.collector.writer=None
            self.assertIsNone(writer.failure);self.assertEqual(len(writer.rows),1)
            self.assertEqual(json.loads((path/'frames.jsonl').read_text())['episode_id'],receipt['episode_id'])
            evidence=json.loads((h.root/'triplet-boundaries/attempt-0000.json').read_text())
            self.assertIn('first_recorded',evidence['phases'])


if __name__=='__main__':unittest.main()
