"""Verified Humble ABI inputs for the collector's borrowed serialized take."""
import hashlib,json,subprocess,shlex
from pathlib import Path

ROS=Path('/opt/ros/humble')
INCLUDES=('rcl','rmw','rcutils','rosidl_runtime_c','rosidl_typesupport_interface','rcl_yaml_param_parser','rcl_logging_interface')
FLAGS=['-std=c11','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror']


def include_flags():return ['-I'+str(ROS/'include'/name) for name in INCLUDES]


def fingerprint(files,compiler):
    raw=json.dumps(dict(files=files,compiler=compiler,flags=FLAGS),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()


def observe():
    from rclpy.impl.implementation_singleton import rclpy_implementation
    import rclpy.executors
    source=Path(__file__).with_name('image_take.c').resolve()
    deps=subprocess.check_output(['cc','-M',*include_flags(),str(source)],text=True,timeout=20)
    headers=[Path(p).resolve() for p in shlex.split(deps.replace('\\\n',' ').split(':',1)[1]) if Path(p).resolve()!=source]
    paths={p:'header' for p in headers}
    maps=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[5].startswith('/'):maps.append(fields[5])
    for name in ('librcl.so','librmw.so','librmw_implementation.so','librmw_cyclonedds_cpp.so','libddsc.so','librcutils.so'):
        candidates={p for p in maps if Path(p).name==name or Path(p).name.startswith(name+'.')}
        if len(candidates)!=1:raise RuntimeError('Native image ABI requires one loaded '+name+': '+repr(candidates))
        path=Path(candidates.pop())
        if not path.is_file():raise RuntimeError('Native image ABI library changed or disappeared: '+str(path))
        paths[path.resolve()]='loaded library'
    paths[Path(rclpy_implementation.__file__).resolve()]='rclpy native'
    for path in list(paths):
        if paths[path]=='header':continue
        linked=subprocess.check_output(['ldd',str(path)],text=True,timeout=20)
        if 'not found' in linked:raise RuntimeError('Unresolved native image ABI library: '+linked)
        for line in linked.splitlines():
            candidate=line.split('=>',1)[-1].strip().split(' ',1)[0]
            if candidate.startswith('/'):
                resolved=Path(candidate).resolve()
                loaded={Path(p).resolve() for p in maps if Path(p).name==resolved.name}
                if loaded!={resolved}:raise RuntimeError('Native image ABI dependency is not the loaded library: '+str(resolved)+' '+repr(loaded))
                paths.setdefault(resolved,'library dependency')
    paths[Path(rclpy.executors.__file__).resolve()]='executor implementation'
    compiler=subprocess.check_output(['cc','--version'],text=True,timeout=10).splitlines()[0]
    files=[dict(path=str(path),kind=kind,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,kind in sorted(paths.items())]
    return dict(version=1,files=files,compiler=compiler,fingerprint=fingerprint(files,compiler))


def verify(observed,expected):
    if expected.get('version')!=1 or expected.get('fingerprint')!=fingerprint(expected['files'],expected['compiler']):
        raise RuntimeError('Malformed native image ABI manifest')
    if observed!=expected:
        previous={p['path']:p for p in expected['files']};current={p['path']:p for p in observed['files']}
        changed=[path for path in sorted(previous.keys()|current.keys()) if previous.get(path)!=current.get(path)]
        raise RuntimeError('Unverified native image ABI; changed dependencies: '+repr(changed)+'; validate this ROS build before updating the checked-in ABI manifest')
    return observed['fingerprint']


def require():
    expected=json.loads(Path(__file__).with_name('image_transport_abi.json').read_text())
    observed=observe();verify(observed,expected)
    return observed


def verify_cache(library,receipt,expected):
    if not library.is_file() or not receipt.is_file():raise RuntimeError('Native image helper cache lacks a complete build receipt')
    row=json.loads(receipt.read_text())
    if (any(row.get(key)!=value for key,value in expected.items()) or row.get('returncode')!=0 or
            row.get('library_sha256')!=hashlib.sha256(library.read_bytes()).hexdigest()):
        raise RuntimeError('Native image helper cache differs from its verified build receipt')
    return row
