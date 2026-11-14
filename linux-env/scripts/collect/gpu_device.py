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


def audit_rows(expected_uuid,owned,rows):
    per_gpu={};per_pid={};observed=[]
    for pid,uuid,memory in rows:
        pid=int(pid)
        if pid not in owned:continue
        amount=int(memory.strip());per_gpu[uuid]=per_gpu.get(uuid,0)+amount
        per_pid.setdefault(str(pid),{})[uuid]=amount
        observed.append(dict(pid=pid,start_ticks=owned[pid]['start_ticks'],gpu_uuid=uuid,memory_mib=amount))
    bad={gpu:amount for gpu,amount in per_gpu.items() if gpu!=expected_uuid and amount>0}
    passed=not bad and per_gpu.get(expected_uuid,0)>0
    return dict(passed=passed,assigned_gpu_uuid=expected_uuid,per_gpu_memory_mib=per_gpu,
                per_pid_memory_mib=per_pid,processes=observed,foreign_gpu_memory_mib=bad)


def owned_gpu_audit(runtime,index):
    from managed_process import verified
    from run_smoke import identity
    expected=gpu_uuid(index);owned={}
    for name in ('isaacsim-soarm101.pid','control.pid'):
        row=verified(Path(runtime)/name)
        if row is None:raise RuntimeError('GPU audit requires a live, verified instance: '+name)
        for old in [row['supervisor'],row.get('workload'),*row.get('children',[])]:
            if old is None:continue
            now=identity(old['pid'])
            if now and now['start_ticks']==old['start_ticks']:owned[old['pid']]=old
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    rows=[[value.strip() for value in row] for row in csv.reader(raw.splitlines()) if row]
    result=audit_rows(expected,owned,rows)
    # Zero on other cards is explicit, including cards with no row for our PIDs.
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    cards={row[0].strip():row[1].strip() for row in csv.reader(inventory.splitlines()) if row}
    for uuid in cards.values():result['per_gpu_memory_mib'].setdefault(uuid,0)
    result['per_gpu_memory_mib_by_index']={index:result['per_gpu_memory_mib'][uuid] for index,uuid in cards.items()}
    result['measurement']='NVML compute-app framebuffer memory for verified PID birth identities; renderer setting is separately recorded'
    result.update(physical_gpu_index=index,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  owned_identities=list(owned.values()))
    result['passed']=result['passed'] and result['cuda_visible_devices']==expected
    return result


if __name__=='__main__':
    import argparse,shlex
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    uuid=gpu_uuid(a.gpu);paths=cache_paths(uuid)
    environment=dict(CUDA_VISIBLE_DEVICES=uuid,SOARM_ASSIGNED_GPU_UUID=uuid,
                     SOARM_CARD_CACHE_ROOT=paths['root'],SOARM_SHADER_CACHE_DIR=paths['shaders'],
                     OPTIX_CACHE_PATH=paths['optix'],CUDA_CACHE_PATH=paths['cuda'])
    for name,value in environment.items():
        if name not in ('CUDA_VISIBLE_DEVICES','SOARM_ASSIGNED_GPU_UUID'):Path(value).mkdir(parents=True,exist_ok=True)
        print('export '+name+'='+shlex.quote(value))
