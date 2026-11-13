import unittest
from types import SimpleNamespace
from checkpoint import progress
from storage import wait_for_storage
from contract import EVAL_SEEDS, corrective_action, parameters, timing_audit


class ContractTests(unittest.TestCase):
    def rows(self):
        out = []
        for i in range(100):
            wall = 1000000000 + round(i * 1e9 / 30)
            sim = 1000000000 + round(i * 1e9 / 30)
            out.append(dict(wall_ns=wall, sim_ns=sim, wrist_stamp_ns=sim, top_stamp_ns=sim,
                            joint_receive_ns=wall - 100, wrist_receive_ns=wall - 200,
                            top_receive_ns=wall - 300, joint_stamp_ns=sim,
                            image_receive_epoch_ns=wall, state=[i * .001] * 6,
                            action=[i * .001 + .035] * 6, executed_action=[i * .001 + .04] * 6, expert_reference=[1. + i * .001] * 6,
                            expert_observation_stamp_ns=sim, command_clock_ns=sim + 33333333))
            out[-1].update(state_source='isaac_articulation', pose_stamp_ns=sim,
                           velocity=[.03] * 6,
                           physics_state=list(out[-1]['state']),physics_velocity=[.03]*6,
                           application_stamp_ns=sim, declared_application_stamp_ns=sim,
                           applied_command=list(out[-1]['executed_action']))
            out[-1]['physics_drive_targets']=list(out[-2]['applied_command'] if i else out[-1]['applied_command'])
        return out

    def test_true_clocks_pass(self):
        self.assertTrue(timing_audit(self.rows(), 30)['passed'])

    def test_five_second_late_application_is_rejected(self):
        rows = self.rows()
        for row in rows:
            row['application_stamp_ns'] += 5000000000
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_reference_feedback_is_rejected(self):
        rows = self.rows(); rows[1]['state_source'] = 'controller_reference'
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_feedback_must_equal_the_independent_physical_snapshot(self):
        rows=self.rows();rows[5]['state'][0]+=.01
        self.assertFalse(timing_audit(rows,30)['passed'])

    def test_drive_overwrite_before_the_next_frame_is_rejected(self):
        rows=self.rows();rows[5]['physics_drive_targets'][0]+=.01
        self.assertFalse(timing_audit(rows,30)['passed'])

    def test_bulk_gate_waits_for_margin_without_gating_root_at_90(self):
        free = iter([49, 51, 52]); waits = []; logs = []
        usage = lambda path: SimpleNamespace(free=(80 if path == '/' else next(free)) * 1024**3)
        stop = SimpleNamespace(wait=lambda seconds: waits.append(seconds) or False)
        wait_for_storage('/bulk', stop, emit=logs.append, usage=usage)
        self.assertEqual(waits, [60, 60])
        self.assertEqual(len(logs), 2)

    def test_near_full_root_still_stops(self):
        with self.assertRaisesRegex(RuntimeError, '20 GiB'):
            wait_for_storage('/bulk', None, usage=lambda path: SimpleNamespace(free=19 * 1024**3))

    def test_resume_ledger_preserves_seed_and_colour(self):
        receipts = []
        for trial, (color, ok) in enumerate([('red', True), ('green', False), ('green', True)]):
            receipts.append(dict(params=dict(parameters(123 + trial, color), trial=trial),
                                 accepted=ok, timing=dict(passed=True)))
        self.assertEqual(progress(receipts, 123), (3, 2))
        with self.assertRaises(ValueError):
            progress(receipts, 124)
        receipts[1]['accepted'] = True
        with self.assertRaises(ValueError):
            progress(receipts, 123)

    def test_duplicate_or_skipped_camera_is_rejected(self):
        rows = self.rows(); rows[30]['sim_ns'] = rows[29]['sim_ns']
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_stale_joint_is_rejected(self):
        rows = self.rows(); rows[40]['joint_stamp_ns'] = 0
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_mislabeled_rate_is_rejected(self):
        self.assertFalse(timing_audit(self.rows(), 20)['passed'])

    def test_expert_corrects_executed_disturbance(self):
        self.assertLess(corrective_action([.02], [0])[0], .02)
        self.assertGreater(corrective_action([-.02], [0])[0], -.02)
        self.assertEqual(corrective_action([0], [1])[0], .035)

    def test_stationary_holds_are_rejected(self):
        rows = self.rows()
        for row in rows:
            row.update(state=[0.] * 6, action=[.035] * 6, expert_reference=[1.] * 6)
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_noisy_labels_are_rejected(self):
        rows = self.rows(); rows[20]['action'] = list(rows[20]['executed_action'])
        self.assertFalse(timing_audit(rows, 30)['passed'])

    def test_seed_reproduction_and_holdout(self):
        self.assertEqual(parameters(123, 'red'), parameters(123, 'red'))
        self.assertEqual(parameters(EVAL_SEEDS[0], 'red', split='eval')['split'], 'eval')
        with self.assertRaises(ValueError):
            parameters(EVAL_SEEDS[0], 'red')


if __name__ == '__main__':
    unittest.main()
