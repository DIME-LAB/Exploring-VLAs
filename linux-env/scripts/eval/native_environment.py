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
