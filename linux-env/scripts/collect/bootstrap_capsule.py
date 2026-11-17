"""Preserve the exact qualified Kit GPU-table bytes independently of live logs."""
import argparse,copy,hashlib,json,os
from pathlib import Path
from storage import confined,atomic_json
from triplet_artifacts import checked_tree,read
from renderer_device import parse_table,resolve
from gpu_device import inventory

ASSIGNED_UUID='GPU-c2772797-ad71-f07e-0101-db0f188f3504'
QUALIFIED_LOG_SHA256='b9ca91d8baacd632229af1d1b678ab949fa91c9f32f9112993ea952797a1f4a5'
SCHEMA='soarm101-rtx-kit-bootstrap-capsule-v1'
KEYS=('passed','assigned_gpu_uuid','kit_index','selected','devices','active_verified')


def prefix(data,expected):
    """Recover only bytes authenticated by the existing qualification hash."""
    digest=hashlib.sha256()
    for index,value in enumerate(data):
        digest.update(bytes((value,)))
        if digest.copy().hexdigest()==expected:return data[:index+1]
    raise ValueError('Historical log contains no exact SHA-matching qualified prefix')


def mapping(pin,data,cards=None):
    if pin.get('passed') is not True or pin.get('assigned_gpu_uuid')!=ASSIGNED_UUID or pin.get('active_verified') is not True:
        raise ValueError('Qualified GPU pin/UUID is missing')
    if pin.get('source',{}).get('sha256')!=QUALIFIED_LOG_SHA256 or hashlib.sha256(data).hexdigest()!=QUALIFIED_LOG_SHA256:
        raise ValueError('Qualified bootstrap bytes changed')
    rows=parse_table(data.decode('utf-8'))
    expected=[{k:r[k] for k in ('kit_index','active','uuid_prefix','pci_bus')} for r in pin['devices']]
    if rows!=expected:raise ValueError('Qualified renderer table differs from its original pin')
    declared=resolve(rows,[dict(uuid=r['physical_uuid'],pci_bus_id=r['pci_bus_id']) for r in pin['devices']],ASSIGNED_UUID,require_active=True)
    if any(declared[k]!=pin[k] for k in KEYS):raise ValueError('Qualified pin is internally inconsistent')
    if cards is not None:
        actual=resolve(rows,cards,ASSIGNED_UUID,require_active=True)
        if len(cards)!=len(pin['devices']) or any(actual[k]!=pin[k] for k in KEYS):
            raise ValueError('Current GPU inventory/mapping differs from the qualified table')
    return rows


def load(path,expected_sha256=None,cards=None):
    path=Path(path)
    if confined(path)!=path:raise ValueError('Capsule path must be canonical')
    checked_tree(path.parent)
    raw=read(path,True);digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Plan-bound bootstrap capsule changed')
    value=json.loads(raw)
    if value.get('schema')!=SCHEMA:raise ValueError('Unqualified bootstrap capsule schema')
    fixture=path.parent/'kit-bootstrap.log'
    if value['fixture']!={'path':str(fixture),'sha256':QUALIFIED_LOG_SHA256,'bytes':fixture.stat().st_size}:
        raise ValueError('Capsule fixture identity differs')
    data=read(fixture,True);mapping(value['qualified_pin'],data,cards)
    pin=copy.deepcopy(value['qualified_pin']);pin['source']=dict(path=str(fixture),sha256=QUALIFIED_LOG_SHA256)
    return dict(pin=pin,capsule_path=str(path),capsule_sha256=digest)


def create(pin_path,output,cards=None):
    pin_path=confined(pin_path);output=confined(output);original=read(pin_path,True);pin=json.loads(original)
    source=Path(pin['source']['path'])
    if confined(source)!=source:raise ValueError('Historical log is not canonical')
    current=read(source,True);qualified=prefix(current,QUALIFIED_LOG_SHA256)
    mapping(pin,qualified,inventory() if cards is None else cards)
    output.mkdir(parents=True,exist_ok=False);fixture=output/'kit-bootstrap.log'
    with fixture.open('xb') as stream:stream.write(qualified);stream.flush();os.fsync(stream.fileno())
    fixture.chmod(0o444)
    capsule=dict(schema=SCHEMA,qualified_pin=pin,
        fixture=dict(path=str(fixture),sha256=QUALIFIED_LOG_SHA256,bytes=len(qualified)),
        recovery=dict(original_pin_path=str(pin_path),original_pin_sha256=hashlib.sha256(original).hexdigest(),
            original_log_path=str(source),original_log_current_sha256=hashlib.sha256(current).hexdigest(),
            exact_qualified_prefix_bytes=len(qualified),appended_bytes=len(current)-len(qualified),
            rule='preserve original qualification hash; no new baseline is admitted'))
    path=output/'capsule.json';atomic_json(path,capsule);path.chmod(0o444)
    result=load(path,cards=inventory() if cards is None else cards)
    if read(source,True)!=current or read(pin_path,True)!=original:raise ValueError('Shared source changed during recovery')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pin',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(create(a.pin,a.output)))


if __name__=='__main__':main()
