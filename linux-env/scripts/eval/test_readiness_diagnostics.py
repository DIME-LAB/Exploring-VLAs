import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from readiness_diagnostics import trace_wait_ready


class Tests(unittest.TestCase):
    def collector(self):
        return SimpleNamespace(node=SimpleNamespace(objects_data={}, _drop_data={}),
                               servo=SimpleNamespace(last_sim_ns=None),
                               sensors=SimpleNamespace(reference={}))

    def test_records_first_observed_conditions_and_restores_original(self):
        collector = self.collector()
        calls = []
        returned = object()

        def original():
            calls.append('original')
            collector.node.objects_data = {'block': 1}
            time.sleep(.015)
            collector.node._drop_data = dict.fromkeys(range(3))
            time.sleep(.015)
            collector.servo.last_sim_ns = 123
            collector.sensors.reference = {'frame': 1}
            return returned

        collector.wait_ready = original
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'readiness.json'
            with trace_wait_ready(collector, path, poll_s=.001):
                self.assertIs(collector.wait_ready(), returned)
            proof = json.loads(path.read_text())
        self.assertIs(collector.wait_ready, original)
        self.assertEqual(calls, ['original'])
        self.assertFalse(proof['budget_changed'])
        self.assertFalse(proof['predicate_changed'])
        self.assertEqual(proof['missing_at_end'], [])
        self.assertEqual(set(proof['first_observed_ready_s']), {'objects_data', 'three_cups', 'servo_last_sim_ns', 'sensors_reference'})
        self.assertLessEqual(proof['first_observed_ready_s']['objects_data'], proof['first_observed_ready_s']['three_cups'])
        self.assertFalse(any(t.name == 'eval-readiness-observer' for t in threading.enumerate()))

    def test_timeout_names_missing_conditions_and_preserves_primary_error(self):
        collector = self.collector()
        collector.sensors.reference = {'frame': 1}
        primary = TimeoutError('original sixty-second budget')

        def original():
            raise primary

        collector.wait_ready = original
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'readiness.json'
            with self.assertRaises(TimeoutError) as raised:
                with trace_wait_ready(collector, path):
                    collector.wait_ready()
            self.assertIs(raised.exception, primary)
            proof = json.loads(path.read_text())
            self.assertEqual(set(proof['missing_at_end']), {'objects_data', 'three_cups', 'servo_last_sim_ns'})
            self.assertEqual(set(proof['first_observed_ready_s']), {'sensors_reference'})
            with patch('storage.atomic_json', side_effect=OSError('disk failure')):
                with self.assertRaises(TimeoutError) as failed:
                    with trace_wait_ready(collector, path):
                        collector.wait_ready()
            self.assertIs(failed.exception, primary)
        self.assertIs(collector.wait_ready, original)


if __name__ == '__main__':
    unittest.main()
