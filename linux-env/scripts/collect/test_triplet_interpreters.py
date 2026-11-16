"""Native source bits govern across Python versions; diagnostics are not commands."""
import json,math,tempfile,unittest
from pathlib import Path
from scene_identity import identity,require_pose_agreement
from triplet_boundary import restore_admission
from target_triplets import validate_run
import test_target_triplets


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


if __name__=='__main__':unittest.main()
