"""Use the collector's qualified host profile for native eval storage."""
import importlib.util,os
from pathlib import Path

def qualified_host(repo):
    path=Path(repo)/'linux-env/scripts/collect/host_config.py'
    spec=importlib.util.spec_from_file_location('native_eval_host_config',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.profile()

def policy_python(host,explicit=None):
    value=explicit or os.environ.get('SOARM_EVAL_POLICY_PYTHON')
    if not value and host['name']=='rtx6000':value='/mnt/storage3/aaugus11/smolvla-train/venv/bin/python'
    if not value:raise ValueError('Set --policy-python to the qualified Torch interpreter on this host')
    # Keep the venv entry path: resolving its symlink to the base executable
    # would discard the venv's package environment on the training host.
    path=Path(value).expanduser().absolute()
    if not path.is_file():raise ValueError('Qualified policy interpreter is missing')
    return str(path)

def preflight_sources(repo):
    """Bind the CPU preflight and its BF collector-side Client import."""
    import hashlib,subprocess
    repo=Path(repo);folder=repo/'linux-env/scripts/eval'
    client=folder/'run_episodes.py'
    original=subprocess.check_output(['git','-C',str(repo),'show','bf7336d:linux-env/scripts/eval/run_episodes.py'])
    if client.read_bytes()!=original:raise ValueError('Native collector-side Client differs from bf7336d')
    names=('native_policy.py','flow_noise.py','grasp_telemetry.py','native_environment.py','native_bank.py','native_replay.py','held_tick.py','checkpoint_contract.py','run_episodes.py')
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}

def require_preflight_sources(proof,repo):
    if proof.get('source_sha256')!=preflight_sources(repo):
        raise ValueError('CPU preflight source differs from the candidate runtime')

def rpc_sources(repo):
    """Fingerprint the actual CPU RPC smoke and production native actor path."""
    import hashlib
    folder=Path(repo)/'linux-env/scripts/eval'
    names=('native_policy_client.py','native_policy_server.py','native_policy_robot.py','flow_noise.py','cpu_rpc_smoke.py','native_environment.py')
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}

def require_rpc_sources(proof,repo):
    if proof.get('source_sha256')!=rpc_sources(repo):
        raise ValueError('CPU RPC proof source differs from the candidate runtime')
