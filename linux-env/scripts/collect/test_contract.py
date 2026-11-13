import unittest
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
                            image_receive_epoch_ns=wall, state=[0.] * 6,
                            action=[.035] * 6, executed_action=[.04] * 6, expert_reference=[1.] * 6,
                            expert_observation_stamp_ns=sim, command_clock_ns=sim + 33333333))
        return out

    def test_true_clocks_pass(self):
        self.assertTrue(timing_audit(self.rows(), 30)['passed'])

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
