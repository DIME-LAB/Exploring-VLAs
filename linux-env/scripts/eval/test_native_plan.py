"""CPU checks for arm selection and socket names before simulation startup."""
import argparse,unittest
from pathlib import Path
from run_native_replay_window import policy_plan
def args(**values):
    return argparse.Namespace(**dict(dict(cadence_diagnostic=False,codec_diagnostic=False,replacement_diagnostic=False,memorisation_diagnostic=False,best_replacement_cadence=None),**values))
class Tests(unittest.TestCase):
    def test_memorisation_matches_default_and_one_selected_replacement(self):
        for cadence in (10,25,50):
            plan=policy_plan(args(memorisation_diagnostic=True,best_replacement_cadence=cadence))
            self.assertEqual(plan,[dict(name='default',cadence=10,rtc=True,aggregation='weighted_average',codec=False),dict(name=f'best{cadence}',cadence=cadence,rtc=False,aggregation='latest_only',codec=False)])
            root=Path('/mnt/storage3/aaugus11/exploring-vlas/eval/runs/native-mem-a-v1')
            self.assertTrue(all(len(str(root/p['name']/'rpc.sock').encode())<104 for p in plan))
    def test_unselected_or_mixed_diagnostics_are_refused(self):
        with self.assertRaises(ValueError):policy_plan(args(memorisation_diagnostic=True))
        with self.assertRaises(ValueError):policy_plan(args(memorisation_diagnostic=True,best_replacement_cadence=10,replacement_diagnostic=True))
    def test_default_and_existing_diagnostics_keep_their_protocol(self):
        self.assertEqual(policy_plan(args()),[dict(name='attempt-01',cadence=10,rtc=True,aggregation='weighted_average',codec=False)])
        self.assertTrue(policy_plan(args(codec_diagnostic=True))[0]['codec'])
        replacement=policy_plan(args(replacement_diagnostic=True))
        self.assertEqual([p['cadence'] for p in replacement],[10,25,50])
        self.assertTrue(all(not p['rtc'] and not p['codec'] and p['aggregation']=='latest_only' for p in replacement))
if __name__=='__main__':unittest.main()
