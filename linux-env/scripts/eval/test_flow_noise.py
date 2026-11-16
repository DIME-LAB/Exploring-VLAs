import json
import subprocess
import sys
import unittest
import torch
from flow_noise import episode_identity, prediction_noise, validate_identity, validate_episode_noise


class PairedNoiseTests(unittest.TestCase):
    def test_deterministic_across_global_rng_and_checkpoint(self):
        identity = episode_identity(530000, 0)
        a, proof = prediction_noise(identity, 10, 1, 50, 32, 'cpu')
        torch.manual_seed(999); torch.randn(1234)
        b, other = prediction_noise(identity, 10, 1, 50, 32, 'cpu')
        self.assertTrue(torch.equal(a, b)); self.assertEqual(proof, other)
        self.assertEqual(a.shape, (1, 50, 32))

    def test_no_global_rng_mutation(self):
        before = torch.get_rng_state().clone()
        prediction_noise(episode_identity(900000, 2), 0, 1, 50, 32, 'cpu')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_repeat_layout_and_step_differ(self):
        ids = [episode_identity(530000, 0), episode_identity(530000, 1), episode_identity(530001, 0)]
        proofs = [prediction_noise(i, s, 1, 50, 32, 'cpu')[1] for i, s in [(ids[0], 0), (ids[1], 0), (ids[2], 0), (ids[0], 10)]]
        self.assertEqual(len({p['noise_sha256'] for p in proofs}), 4)

    def test_invalid_identity_refused(self):
        for value in (None, {}, dict(episode_identity(1, 0), episode_seed=3)):
            with self.assertRaises(ValueError): validate_identity(value)
        for value in (-1, 1.0, True, '1'):
            with self.assertRaises(ValueError): episode_identity(value, 0)

    def test_fresh_process_same_hash(self):
        script = 'from flow_noise import *; import json; print(json.dumps(prediction_noise(episode_identity(530000,0),10,1,50,32,"cpu")[1]))'
        proof = prediction_noise(episode_identity(530000, 0), 10, 1, 50, 32, 'cpu')[1]
        self.assertEqual(proof, json.loads(subprocess.check_output([sys.executable, '-c', script], text=True)))

    def test_parent_refuses_dropped_or_changed_rpc_proofs(self):
        import copy
        identity = episode_identity(530000, 0)
        proof = prediction_noise(identity, 0, 1, 50, 32, 'cpu')[1]
        item = dict(step=0, native_rtc=dict(flow_noise=proof))
        good = dict(flow_noise=identity, requests=[item], chunks=[copy.deepcopy(item)])
        validate_episode_noise(good, identity, 50, 32)
        for field in ('shape', 'chunk_seed', 'action_step', 'episode_seed', 'noise_sha256'):
            broken = copy.deepcopy(good); broken['chunks'][0]['native_rtc']['flow_noise'][field] = None
            with self.assertRaises(ValueError): validate_episode_noise(broken, identity, 50, 32)
        broken = copy.deepcopy(good); broken['requests'][0]['native_rtc'] = {}
        with self.assertRaises(ValueError): validate_episode_noise(broken, identity, 50, 32)
        with self.assertRaises(ValueError): validate_episode_noise(good, None, 50, 32)
        validate_episode_noise(dict(flow_noise=None, requests=[dict(step=0)], chunks=[dict(step=0)]), None, 50, 32)

    def server_fixture(self, identity=None):
        import os
        import types
        from unittest.mock import patch
        import native_policy_server
        production = types.ModuleType('lerobot.async_inference.policy_server')
        production.get_policy_class = lambda name: object
        class Base:
            def _predict_action_chunk(self, obs):
                self._get_action_chunk(obs)
                return [types.SimpleNamespace()]
        production.PolicyServer = Base
        package = types.ModuleType('lerobot.async_inference'); package.policy_server = production
        with patch.dict(sys.modules, {'lerobot': types.ModuleType('lerobot'), 'lerobot.async_inference': package, 'lerobot.async_inference.policy_server': production}), patch.dict(os.environ, {'SOARM_EVAL_NATIVE_HELD': '1', 'SOARM_EVAL_TORCH_MEMORY_GIB': ''}):
            native_policy_server.install()
        server = production.PolicyServer()
        context = dict(mode='collect', held=True, version=1, sim_ns=123, epoch=4)
        if identity is not None: context['flow_noise'] = identity
        server.last_processed_obs = types.SimpleNamespace(get_observation=lambda: {'_eval_native_held': context}, get_timestep=lambda: 10)
        server._rtc_previous_chunk = None; server._rtc_chunk_start = 0; server.actions_per_chunk = 50
        server.calls = []
        def predict(obs, **kwargs):
            server.calls.append(kwargs)
            return kwargs.get('noise', torch.zeros(1, 50, 32))[:, :, :6]
        server.policy = types.SimpleNamespace(config=types.SimpleNamespace(rtc_config=None, chunk_size=50, max_action_dim=32), predict_action_chunk=predict)
        return server

    def test_actual_server_default_and_seeded_rpc_metadata(self):
        observation = {'observation.state': torch.zeros(1, 6)}
        plain = self.server_fixture()
        plain_result = plain._predict_action_chunk(observation)
        self.assertEqual(plain.calls, [{}]); self.assertNotIn('flow_noise', plain_result[0].native_rtc)
        identity = episode_identity(530000, 0)
        seeded = self.server_fixture(identity)
        result = seeded._predict_action_chunk(observation)
        expected, proof = prediction_noise(identity, 10, 1, 50, 32, 'cpu')
        self.assertTrue(torch.equal(seeded.calls[0]['noise'], expected))
        self.assertEqual(result[0].native_rtc['flow_noise'], proof)

    def test_actual_server_preserves_rtc_tail_and_held_guard(self):
        import types
        server = self.server_fixture(episode_identity(530000, 1))
        server.policy.config.rtc_config = types.SimpleNamespace(enabled=True, execution_horizon=10, max_guidance_weight=10)
        server._rtc_previous_chunk = torch.ones(1, 50, 6); server._rtc_chunk_start = 0
        server._get_action_chunk({'observation.state': torch.zeros(1, 6)})
        self.assertTrue(torch.equal(server.calls[0]['prev_chunk_left_over'], torch.ones(1, 40, 6)))
        self.assertEqual(server.calls[0]['inference_delay'], 0)
        server.last_processed_obs.get_observation()['_eval_native_held']['held'] = False
        with self.assertRaises(RuntimeError): server._get_action_chunk({'observation.state': torch.zeros(1, 6)})


if __name__ == '__main__': unittest.main()
