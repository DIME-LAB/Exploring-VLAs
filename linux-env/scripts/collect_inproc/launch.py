"""Preflight or launch one owned standalone instance; requires a lead sim window."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shared
from storage import confined,prepare_environment
from gpu_device import gpu_uuid,inventory,cache_paths
from renderer_device import read_table,resolve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['preflight','start','stop','status'])
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--python',type=Path,default=Path.home()/'env_isaacsim61/bin/python')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--renderer-log',type=Path,help='Prior verified Kit GPU table for renderer-index resolution')
    p.add_argument('--window',help='Explicit lead-granted sim window; required to start')
    args,rest=p.parse_known_args()
    runtime=confined(args.runtime)
    from managed_process import start,stop,verified
    pidfile=runtime/'inprocess.pid'
    if args.operation=='stop':print(json.dumps(stop(pidfile)));return
    if args.operation=='status':print(json.dumps(verified(pidfile)));return
    if not args.python.is_file():raise ValueError('Isaac interpreter not found')
    if not args.renderer_log:raise ValueError('Prior renderer table log required; CUDA visibility does not pin Vulkan')
    uuid=gpu_uuid(args.gpu);rows,source=read_table(args.renderer_log)
    renderer=resolve(rows,inventory(),uuid)
    if not rest or rest[0]!='--':raise ValueError('Pass run.py arguments after --')
    from run import arguments
    parsed=arguments([*rest[1:],'--window',args.window or 'PENDING'])
    missing=[str(path) for path in [parsed.scene,parsed.manifest,*parsed.fixture] if not path.is_file()]
    if parsed.output.exists():missing.append('Output already exists: '+str(parsed.output))
    command=[str(args.python),'-u',str(Path(__file__).with_name('run.py')),*rest[1:]]
    info=dict(command=command,assigned_gpu_uuid=uuid,renderer=renderer,prior_log=source,
        caches=cache_paths(uuid),runtime=str(runtime),window=args.window,
        inputs_ready=not missing,missing_inputs=missing)
    # Preflight prints a reviewable command and never imports or launches Isaac.
    if args.operation=='preflight':print(json.dumps(info,indent=2));return
    if not args.window:raise ValueError('Starting Isaac requires an explicit sim window from the lead')
    if missing:raise ValueError('Required inputs are not ready: '+repr(missing))
    command.extend(['--window',args.window])
    runtime.mkdir(parents=True,exist_ok=True)
    os.environ.update(SOARM_INSTANCE_ROOT=str(runtime),SOARM_ASSIGNED_GPU_UUID=uuid,
        CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        SOARM_RENDERER_GPU_INDEX=str(renderer['kit_index']),
        SOARM_CURRENT_KIT_LOG=str(runtime/'kit.log'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    prepare_environment(31)
    # Persistent per-card caches, including Kit's shader and derived-data roots.
    card=cache_paths(uuid)
    os.environ['XDG_CACHE_HOME']=card['root']
    watched=[sys.executable,str(Path(__file__).with_name('watch.py')),'--',*command]
    print(json.dumps(dict(info,supervisor_pid=start(pidfile,runtime/'console.log',watched)),indent=2))


if __name__=='__main__':main()
