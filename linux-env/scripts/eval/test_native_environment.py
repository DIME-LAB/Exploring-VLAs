import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from native_environment import policy_python,preflight_sources,require_preflight_sources,rpc_sources,require_rpc_sources
class Tests(unittest.TestCase):
    def test_venv_entry_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);base=root/'base-python';base.write_text('binary')
            entry=root/'venv/bin/python';entry.parent.mkdir(parents=True);entry.symlink_to(base)
            self.assertEqual(policy_python({'name':'dual-a4500'},entry),str(entry))
            self.assertNotEqual(policy_python({'name':'dual-a4500'},entry),str(base))
    def test_preflight_refuses_changed_source_or_wrong_collector_client(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'linux-env/scripts/eval';folder.mkdir(parents=True)
            for name in ('native_policy.py','flow_noise.py','grasp_telemetry.py','native_environment.py','native_bank.py','native_replay.py','held_tick.py','checkpoint_contract.py','run_episodes.py','rule_target.py','prepare_seen15_bank.py','wrist_retention.py','native_policy_bridge.py','closure_geometry.py','readiness_diagnostics.py'):(folder/name).write_text(name)
            with patch('subprocess.check_output',return_value=b'run_episodes.py'):
                proof={'source_sha256':preflight_sources(root)}
                require_preflight_sources(proof,root)
                for name in ('native_policy.py','flow_noise.py','grasp_telemetry.py'):
                    (folder/name).write_text('changed')
                    with self.assertRaisesRegex(ValueError,'source differs'):require_preflight_sources(proof,root)
                    (folder/name).write_text(name)
                (folder/'run_episodes.py').write_text('different client')
                with self.assertRaisesRegex(ValueError,'Client differs'):preflight_sources(root)
    def test_rpc_proof_refuses_changed_production_actor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'linux-env/scripts/eval';folder.mkdir(parents=True)
            for name in ('native_policy_client.py','native_policy_server.py','native_policy_robot.py','flow_noise.py','cpu_rpc_smoke.py','native_environment.py'):(folder/name).write_text(name)
            proof={'source_sha256':rpc_sources(root)}
            require_rpc_sources(proof,root)
            for name in ('native_policy_server.py','flow_noise.py'):
                (folder/name).write_text('changed')
                with self.assertRaisesRegex(ValueError,'RPC proof source differs'):require_rpc_sources(proof,root)
                (folder/name).write_text(name)
    def test_dual_does_not_guess_an_interpreter(self):
        with patch.dict(os.environ,{'SOARM_EVAL_POLICY_PYTHON':''}):
            with self.assertRaisesRegex(ValueError,'qualified Torch interpreter'):policy_python({'name':'dual-a4500'})
if __name__=='__main__':unittest.main()
