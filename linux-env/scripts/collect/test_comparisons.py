"""Retained native snapshots through real capture, row, proof and physical-verdict paths."""
import copy,json,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from comparisons import comparison,matches,native_frame
from contract import timing_audit
from protocol import StepGeneration
from success import initial_arm,settled_containment
from tick_contract import TickContract,TickContractError,require_snapshots,verify_proof,verdict_contract_valid,digest
from test_restore_clock import capture_snapshot
from test_tick_contract import checked_window
import test_review_regressions as probes

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('retained_verdict_snapshots.json').read_text())


class ComparisonTests(unittest.TestCase):
    def test_all_26_retained_native_snapshots_pass_capture_and_each_verdict_contract_with_camera_skew(self):
        for item in FIXTURE['snapshots']:
            with self.subTest(trial=item['trial'],source=item['source_snapshot']):
                snapshot=item['snapshot'];c,r,entry=probes.ReviewRegressions().collector()
                epoch=snapshot['epoch'];stamp=snapshot['sim_ns'];c.generation=StepGeneration()
                c.generation.install(dict(epoch=epoch,invalid_through_ns=stamp-33333333))
                r.epoch=epoch;r.invalid_through_ns=stamp-33333333
                c.servo.generation=c.generation;c.servo.plan=None;c.servo.reference=list(snapshot['state']);c.servo.reference_epoch=epoch
                r.art.get_applied_actions().joint_positions[:]=[snapshot['drive_targets']]
                capture_snapshot(c,r,entry,snapshot,camera_offset_ns=2)
                self.assertIsNone(c.failure);held=c.last_snapshot;proof=held['tick_contract']
                self.assertEqual(proof['row']['sim_ns']-held['sim_ns'],2)
                self.assertEqual(verify_proof(proof),[])
                contract=require_snapshots([held]);self.assertTrue(verdict_contract_valid(contract))
                self.assertTrue(initial_arm([held],snapshot['state'][:5])['contract_passed'])
                self.assertTrue(settled_containment([held],'red_2x3_0','red')['contract_passed'])
                audit=timing_audit([proof['row']],30)
                self.assertEqual(audit['contract_errors'],['Planning/hold frames exceed 3%'])
                self.assertTrue(timing_audit([dict(proof['row'],executing=True)],30)['contract_passed'])
                # Native recovery endpoints reconstruct exactly, independent of camera stamps.
                expected={k:held[k] for k in ('epoch','sim_ns','state_source','state','bodies')}
                self.assertTrue(matches(native_frame(proof['row']),expected))
                # Restore-baseline identity stays copied/exact; both clocks retain their origin.
                boundary=TickContract();boundary.previous=copy.deepcopy(proof['row'])
                native=dict(epoch=epoch+1,sim_ns=stamp,physics_time_s=snapshot['physics_time_s'],drive_targets=snapshot['drive_targets'])
                restored=dict(snapshot=dict(snapshot,epoch=epoch+1),epoch=epoch+1,invalid_through_ns=stamp,
                              command_baseline=native,arm_restored=True)
                boundary.install_boundary(restored,restore=True)

    def test_two_ns_camera_offset_passes_but_changed_native_copy_does_not(self):
        held=checked_window([{}])[0];row=held['tick_contract']['row']
        for key in ('sim_ns','wrist_stamp_ns','top_stamp_ns','expert_observation_stamp_ns','observation_stamp_ns',
                    'declared_application_stamp_ns','command_clock_ns'):
            row[key]+=2
        row['execution_geometry']['sim_ns']+=2
        held['tick_contract']['row_sha256']=digest(row)
        self.assertTrue(require_snapshots([held])['passed'])
        held['sim_ns']+=1
        with self.assertRaises(TickContractError) as rejected:require_snapshots([held])
        difference=rejected.exception.evidence['contract']['comparisons'][0]
        self.assertEqual(difference['field'],'snapshot.sim_ns vs row.pose_stamp_ns')
        self.assertEqual(difference['rule'],'copy');self.assertEqual(difference['delta'],1)
        self.assertIn('snapshots',rejected.exception.evidence)

    def test_producer_tolerance_does_not_admit_a_microsecond_gap_or_large_state_change(self):
        self.assertTrue(matches(10**12+1000,10**12,'stamp_ns'))
        self.assertFalse(matches(10**12+1001,10**12,'stamp_ns'))
        self.assertFalse(matches(10**12+33333333,10**12,'stamp_ns'))
        self.assertFalse(matches([.1],[.10001],'state_rad'))
        self.assertFalse(matches(float('nan'),float('nan'),'world_s'))

    def test_recovery_native_endpoint_binding_survives_camera_rounding(self):
        import test_recovery as recovery_probes
        from contract import parameters
        from recovery import receipt_error
        c,_=recovery_probes.RecoveryTests().driver();params=parameters(1,'red',regrasp=True);report={}
        c.regrasp_prefix(params,1,report)
        rows=copy.deepcopy(c.test_rows)
        for row in rows:
            for key in ('sim_ns','wrist_stamp_ns','top_stamp_ns','expert_observation_stamp_ns',
                        'observation_stamp_ns','declared_application_stamp_ns','command_clock_ns'):
                row[key]+=2
            row['execution_geometry']['sim_ns']+=2
        recovery=report['recovery_report'];before=recovery['measured_before']['sim_ns'];after=recovery['measured_after_lift']['sim_ns']
        recovery['window_frame_sha256s']=[digest(r) for r in rows if before<=native_frame(r)['sim_ns']<=after]
        receipt=dict(params=params,episode_id=c.current_episode_id,**report)
        self.assertIsNone(receipt_error(receipt,rows))
        next(row for row in rows if native_frame(row)['sim_ns']==before)['pose_stamp_ns']+=1
        self.assertIsNotNone(receipt_error(receipt,rows))
