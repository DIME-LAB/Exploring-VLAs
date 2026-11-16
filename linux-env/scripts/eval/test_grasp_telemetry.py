import unittest
import numpy as np
from grasp_telemetry import ContactInterval, aperture_at_depth, line_box_interval, pose_matrix, install, guarded_snapshot, validate_intervals


class GeometryTests(unittest.TestCase):
    def box(self, x):
        matrix = np.eye(4); matrix[0, 3] = x
        return dict(world_matrix=matrix.tolist(), half_size_m=[.002, .01, .02])

    def test_finite_aperture_and_missing_depth(self):
        result = aperture_at_depth([0, 0, 0], [1, 0, 0], [self.box(-.012)], [self.box(.012)])
        self.assertEqual(result['status'], 'measured-finite-pad-line')
        self.assertAlmostEqual(result['signed_separation_m'], .02)
        missing = aperture_at_depth([0, .02, 0], [1, 0, 0], [self.box(-.012)], [self.box(.012)])
        self.assertIsNone(missing['signed_separation_m'])
        self.assertEqual(missing['status'], 'unavailable-outside-finite-pad-depth')

    def test_rotation_and_scale_are_applied(self):
        rotation = pose_matrix([.1, .2, .3], [2**-.5, 0, 0, 2**-.5])
        matrix = rotation @ np.diag([2, 1, 1, 1])
        result = line_box_interval([.1, .2, .3], [0, 1, 0], matrix, [.002, .01, .02])
        self.assertAlmostEqual(result[0], -.004); self.assertAlmostEqual(result[1], .004)

    def test_no_implicit_installation(self):
        class Fake: pass
        with self.assertRaises(RuntimeError): install(Fake())


class ContactTests(unittest.TestCase):
    def test_telemetry_exception_does_not_poison_physics_snapshot(self):
        stream = ContactInterval()
        def original(): return dict(epoch=1, sim_ns=0, state=[1, 2, 3])
        def broken(result): raise ValueError('forced pose error')
        wrapped = guarded_snapshot(original, broken, stream)
        result = wrapped()
        self.assertEqual(result['state'], [1, 2, 3])
        self.assertEqual(result['grasp_telemetry']['status'], 'unavailable')
        with self.assertRaises(ValueError): validate_intervals([{}, result])
        self.assertEqual(wrapped()['state'], [1, 2, 3])

    def test_idempotent_owned_install_and_changed_source_refusal(self):
        import hashlib
        import types
        from pathlib import Path
        import grasp_telemetry
        stream = ContactInterval(); wrapper = lambda: None
        runtime = types.SimpleNamespace(mode='collect', timeline=types.SimpleNamespace(is_playing=lambda: False), stage=object(), snapshot=wrapper)
        runtime._eval_grasp_telemetry = dict(source_sha256=hashlib.sha256(Path(grasp_telemetry.__file__).read_bytes()).hexdigest(), stage=runtime.stage, wrapper=wrapper, interval=stream, step_subscription=object(), contact_subscription=object(), proof=dict(enabled=True, version=1))
        self.assertTrue(install(runtime, enabled=True)['reused'])
        runtime._eval_grasp_telemetry['source_sha256'] = 'changed'
        with self.assertRaises(RuntimeError): install(runtime, enabled=True)

    def test_interval_acceptance_requires_full_physics_duration(self):
        stream = ContactInterval(); stream.drain(1, 0); stream.step(1/30, 33333333)
        valid = dict(grasp_telemetry=dict(contacts=stream.drain(1, 33333333)))
        self.assertEqual(validate_intervals([{}, valid]), dict(contact_ticks=0, zero_contact_ticks=1, physics_substeps=1))
        valid['grasp_telemetry']['contacts']['physics_dt_sum_s'] = 1/60
        with self.assertRaises(ValueError): validate_intervals([{}, valid])

    def test_aggregates_all_substeps_and_keeps_held_snapshot(self):
        stream = ContactInterval()
        self.assertEqual(stream.drain(1, 0)['status'], 'unavailable')
        for tick in range(3):
            stream.step(1/90, round((tick+1)*1e9/90))
            stream.add(dict(points=[dict(impulse_Ns=[tick+1, 0, 0])], friction_anchors=[dict(impulse_Ns=[0, .25, 0])]))
        result = stream.drain(1, 33333333)
        self.assertEqual(result['physics_substep_count'], 3)
        self.assertAlmostEqual(result['physics_dt_sum_s'], 1/30)
        self.assertEqual(result['contact_impulse_vector_sum_Ns'], [6, 0, 0])
        self.assertEqual(result['friction_anchor_impulse_vector_sum_Ns'], [0, .75, 0])
        self.assertIs(stream.drain(1, 33333333), result)
        stream.step(1/30, 66666666)
        self.assertEqual(stream.drain(1, 66666666)['status'], 'zero-contact')

    def test_unavailable_distinct_from_zero(self):
        stream = ContactInterval(); stream.drain(1, 0)
        self.assertEqual(stream.drain(1, 1)['status'], 'unavailable')
        stream.step(.1, 2); self.assertEqual(stream.drain(2, 2)['status'], 'unavailable')
        stream.error = 'callback failure'; stream.step(.1, 3)
        self.assertEqual(stream.drain(2, 3)['status'], 'unavailable')


if __name__ == '__main__': unittest.main()
