"""Fingerprint loaded local Python sources and compare native recording inputs."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys


def source_digest(path):
    try:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:raise RuntimeError('Provenance source is unavailable: '+str(path)) from exc


def local_path(directory,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.py':
        raise RuntimeError('Invalid local Python provenance name: '+str(name))
    path=(Path(directory)/relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):raise RuntimeError('Provenance source leaves its directory: '+str(name))
    return path


def verify_sources(directory,expected):
    for name,digest in expected.items():
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):
            raise RuntimeError('Invalid source fingerprint: '+str(name))
        actual=source_digest(local_path(directory,name))
        if actual!=digest:raise RuntimeError('Loaded provenance source changed: '+str(name)+'; expected='+digest+'; observed='+actual)


def loaded_sources(directory,previous=None,modules=None):
    """Keep first fingerprints; refuse changed/deleted sources, including unloaded ones."""
    directory=Path(directory).resolve();result=dict(previous or {})
    verify_sources(directory,result)
    modules=dict(sys.modules) if modules is None else dict(modules)
    for module_name,module in modules.items():
        # Inspect stored metadata without invoking unrelated lazy modules'
        # __getattr__ hooks (common in the Kit/Torch process).
        filename=getattr(module,'__dict__',{}).get('__file__')
        if not filename:continue
        raw=Path(filename).absolute()
        if raw.suffix=='.pyc':
            try:raw=Path(importlib.util.source_from_cache(str(raw)))
            except ValueError:
                if raw.is_relative_to(directory):raise RuntimeError('Unknown local bytecode source: '+str(raw))
                continue
        resolved=raw.resolve()
        if raw.is_relative_to(directory) and not resolved.is_relative_to(directory):
            raise RuntimeError('Loaded local source leaves its directory: '+str(raw))
        if ('.' not in module_name and (directory/(module_name+'.py')).is_file() and
                resolved.parent!=directory):
            raise RuntimeError('Local Python module loaded from another directory: '+module_name+'; '+str(raw))
        if not resolved.is_relative_to(directory) or resolved.suffix!='.py':continue
        name=resolved.relative_to(directory).as_posix();digest=source_digest(resolved)
        if name in result and result[name]!=digest:raise RuntimeError('Loaded provenance source changed: '+name)
        result[name]=digest
    return result


def recording_manifest(repo,explicit,native_dependencies,collector_loaded=None,modules=None):
    """Native-only sources join the manifest even when the recorder never imports them."""
    repo=Path(repo).resolve();directory=repo/'linux-env/scripts/collect'
    local=loaded_sources(directory,collector_loaded,modules)
    verify_sources(directory,native_dependencies)
    names=set(explicit)|{'linux-env/scripts/collect/'+n for n in set(local)|set(native_dependencies)}
    result={}
    for name in sorted(names):
        path=(repo/name).resolve()
        if not path.is_relative_to(repo):raise RuntimeError('Recording source leaves repository: '+name)
        result[name]=source_digest(path)
    return result
