"""Validity/selection regressions: bad dispatch must never become a losing model score."""
import json,tempfile,unittest
from pathlib import Path
from report_phase3 import validity,report

class ReportTests(unittest.TestCase):
    def row(self):
        return dict(outcome='no-grasp',clock_basis='simulation',sim_cap_hz=30,client=dict(
            clock_basis='simulation',missed_action_ticks_after_startup=0,nonmonotonic_action_steps=0,
            send_action_costs_ms=[.4]*30,observe_every_actions=10,clock_dispatch_trace=[dict(clock_s=1)],
            observation_cadence_events=[dict(executed_actions=0),dict(executed_actions=10,actions_since_request=10,empty_slots_since_request=0)]))
    def test_queue_starvation_is_an_outcome_not_an_exclusion(self):
        row=self.row();row['empty_ticks_pct']=80;row['effective_action_fps']=6
        self.assertEqual(validity(row),[])
    def test_dispatch_failure_never_scores(self):
        row=self.row();row['client']['missed_action_ticks_after_startup']=1
        self.assertIn('missed-sim-slots',validity(row))
        row=self.row();row['client']['send_action_costs_ms']=[30.]*30
        self.assertIn('send-cost',validity(row))
    def test_incomplete_set_cannot_win(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(report(Path(directory))['best_checkpoint'])

if __name__=='__main__':unittest.main(verbosity=2)
