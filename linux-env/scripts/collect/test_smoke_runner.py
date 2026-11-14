"""Frozen smoke recipe and exact-process cleanup; no simulator launches."""
import unittest
from pathlib import Path
from unittest.mock import patch
import run_smoke


class SmokeRunnerTests(unittest.TestCase):
    def test_recipe_has_normal_and_dedicated_regrasp_with_diagnostic_retention(self):
        recipe=run_smoke.plan('/mnt/a4500-backup/exploring-vlas/datasets/unit-future-smoke-not-created','a'*40)
        self.assertEqual(recipe['domain'],90);self.assertEqual(recipe['bridge_port'],8777)
        commands=[s['argv'] for s in recipe['stages'] if '-first' in s['name'] or '-resume' in s['name']]
        self.assertEqual(len(commands),4)
        self.assertTrue(all('--diagnostic' in c and c[c.index('--episodes')+1]=='5' for c in commands))
        self.assertEqual(sum('--regrasp-only' in c for c in commands),2)
        self.assertFalse(recipe['pilot_started'])

    def test_foreign_output_or_nonassigned_instance_refuses(self):
        with self.assertRaises(ValueError):run_smoke.plan('/tmp/not-authorized','a'*40)
        with self.assertRaises(ValueError):run_smoke.plan('/mnt/a4500-backup/exploring-vlas/datasets/unit-unused-smoke','a'*40,1)

    def test_source_change_refuses_before_any_stage(self):
        with patch('run_smoke.subprocess.check_output',side_effect=['collect.py\n','']):
            with self.assertRaisesRegex(RuntimeError,'Frozen smoke source'):run_smoke.frozen('a'*40)

    def test_reused_pid_never_authorizes_a_signal(self):
        owned=run_smoke.OwnedProcesses(Path('/not-used'),0)
        owned.known[123]=dict(pid=123,start_ticks='old')
        newer=dict(pid=123,start_ticks='new',ppid=1,state='S')
        with patch('run_smoke.identity',return_value=newer),patch('run_smoke.os.kill') as kill:
            self.assertEqual(owned.remaining(),[])
            owned.stop();kill.assert_not_called()
