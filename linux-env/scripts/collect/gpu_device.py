"""One physical GPU per instance; read-only NVML evidence, never CUDA discovery."""
import csv,json,os,re,socket,subprocess
from pathlib import Path
from storage import CANONICAL_ROOT,confined


class GpuIsolationError(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('Assigned GPU isolation failed: '+json.dumps(evidence,sort_keys=True))


def gpu_uuid(index):
    value=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(index)],text=True).strip()
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',value):raise RuntimeError('Assigned GPU UUID is unavailable: '+value)
    return value


def cache_paths(uuid,hostname=None):
    host=hostname or socket.gethostname()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',host) or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',uuid):raise ValueError('Invalid device cache identity')
    root=confined(CANONICAL_ROOT/'cache/isaac'/host/uuid)
    return dict(root=str(root),shaders=str(root/'shaders'),optix=str(root/'optix'),cuda=str(root/'cuda'))


def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader'],text=True)
    return [dict(index=int(row[0]),uuid=row[1].strip(),pci_bus_id=row[2].strip()) for row in csv.reader(raw.splitlines()) if row]


def owned_snapshot(runtime):
    from managed_process import verified
    from run_smoke import identity
    roots=[]
    for name in ('isaacsim-soarm101.pid','control.pid'):
        row=verified(Path(runtime)/name)
        if row is None:raise RuntimeError('GPU audit requires a live, verified instance: '+name)
        roots.append(row['supervisor'])
    own=identity(os.getpid())
    if own:roots.append(own)
    found={};todo=list(roots)
    while todo:
        old=todo.pop();now=identity(old['pid'])
        if now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ('pid','start_ticks','pgid')):
            raise RuntimeError('Owned process birth changed during GPU attribution')
        if now['pid'] in found:continue
        found[now['pid']]=now
        # Only kernel ancestry rooted at verified supervisors/this collector is used.
        for task in Path('/proc',str(now['pid']),'task').iterdir():
            try:children=(task/'children').read_text().split()
            except FileNotFoundError:continue
            for pid in children:
                child=identity(int(pid));parent=identity(now['pid'])
                if parent is None or parent['start_ticks']!=now['start_ticks']:raise RuntimeError('Owned parent changed')
                if child and child['state']!='Z' and child['ppid']==now['pid']:todo.append(child)
    return found


def birth_set(owned):return {(r['pid'],r['start_ticks'],r['pgid']) for r in owned.values()}


def audit_sample(expected,owned,cards,sample):
    coverage=sample['coverage'];errors=list(sample['errors']);per_pid={};contexts=[]
    for card in cards:
        if any(coverage.get(card['uuid'],{}).get(api)!='verified' for api in ('compute','graphics')):
            errors.append(dict(gpu_uuid=card['uuid'],error='Compute/graphics attribution unavailable'))
    for row in sample['rows']:
        if row['pid'] not in owned:continue
        contexts.append(dict(row,start_ticks=owned[row['pid']]['start_ticks']))
        if row['memory_bytes'] is None:
            errors.append(dict(row,error='Owned process GPU memory is unavailable'));continue
        pair=(row['pid'],row['gpu_uuid'])
        per_pid[pair]=max(per_pid.get(pair,0),row['memory_bytes'])
    # Compute/graphics APIs report overlapping per-process allocations: max per
    # PID/card, not their sum. Even a zero-memory foreign context is a violation.
    foreign=[r for r in contexts if r['gpu_uuid']!=expected]
    verified=not errors
    per_gpu={card['uuid']:sum(v for (pid,uuid),v in per_pid.items() if uuid==card['uuid'])/1024**2 if verified else None for card in cards}
    passed=verified and not foreign and bool(per_gpu.get(expected,0))
    return dict(passed=passed,outcome='verified' if verified else 'unverified',retryable=False,
        assigned_gpu_uuid=expected,per_gpu_memory_mib=per_gpu,
        per_gpu_memory_mib_by_index={str(c['index']):per_gpu[c['uuid']] for c in cards},
        coverage=coverage,processes=contexts,errors=errors,foreign_gpu_contexts=foreign,
        measurement='NVML compute + graphics queries; maximum shared per-PID/card memory avoids double counting')


def owned_gpu_audit(runtime,index):
    from nvml_processes import sample
    from renderer_device import read_table,resolve
    attempts=[]
    for _ in range(3):
        try:
            cards=inventory();expected=next(c['uuid'] for c in cards if c['index']==index)
            before=owned_snapshot(runtime);observed=sample(cards);after=owned_snapshot(runtime)
            if birth_set(before)!=birth_set(after):raise RuntimeError('Owned births changed across NVML sampling')
            result=audit_sample(expected,before,cards,observed)
            rows,source=read_table(Path(runtime)/'isaac.log')
            result['renderer']=dict(resolve(rows,cards,expected,require_active=True),source=source)
            result.update(physical_gpu_index=index,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                owned_identities=list(before.values()),births_checked_before_and_after=True,sampling_attempts=attempts)
            result['passed']=result['passed'] and result['cuda_visible_devices']==expected
            return result
        except Exception as exc:
            attempts.append(dict(error=repr(exc),outcome='unverified'))
    return dict(passed=False,outcome='unverified',retryable=True,sampling_attempts=attempts,
                error='GPU attribution/renderer identity could not be verified; no zero inferred')


if __name__=='__main__':
    import argparse,shlex
    from renderer_device import read_table,resolve
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--kit-log',type=Path);p.add_argument('--renderer-index',action='store_true')
    p.add_argument('--verify-renderer',action='store_true');a=p.parse_args()
    uuid=gpu_uuid(a.gpu);paths=cache_paths(uuid)
    renderer=None
    if a.kit_log:
        rows,source=read_table(a.kit_log);renderer=dict(resolve(rows,inventory(),uuid,require_active=a.verify_renderer),source=source)
    if a.renderer_index:
        if renderer is None:raise RuntimeError('Renderer requires a physical Kit device map')
        print(renderer['kit_index']);raise SystemExit(0)
    if a.verify_renderer:
        if renderer is None:raise RuntimeError('Renderer identity unverified')
        print(json.dumps(renderer));raise SystemExit(0)
    environment=dict(CUDA_VISIBLE_DEVICES=uuid,SOARM_ASSIGNED_GPU_UUID=uuid,
                     SOARM_CARD_CACHE_ROOT=paths['root'],SOARM_SHADER_CACHE_DIR=paths['shaders'],
                     OPTIX_CACHE_PATH=paths['optix'],CUDA_CACHE_PATH=paths['cuda'])
    if renderer:environment['SOARM_RENDERER_GPU_INDEX']=str(renderer['kit_index'])
    for name,value in environment.items():
        if name not in ('CUDA_VISIBLE_DEVICES','SOARM_ASSIGNED_GPU_UUID','SOARM_RENDERER_GPU_INDEX'):Path(value).mkdir(parents=True,exist_ok=True)
        print('export '+name+'='+shlex.quote(value))
