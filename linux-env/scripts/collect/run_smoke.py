#!/usr/bin/env python3
"""Frozen normal + re-grasp smoke, using only the assigned private instance.

--plan prints commands without starting processes. --go is for an explicit lead
smoke window; independent review may run concurrently and can supersede the data.
"""
import argparse,ctypes,fcntl,json,os,re,select,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from storage import CANONICAL_ROOT,confined

REPO=Path(__file__).resolve().parents[3]
SCOPES=('linux-env/scripts/collect','linux-env/scripts/collect_sim.sh','linux-env/scripts/isaac/isaacsim_launch.sh',
        'vla_SO-ARM101/src/so_arm101_control','vla_SO-ARM101/src/so_arm101_moveit_config','vla_SO-ARM101/src/so_arm101_description','isaac/exts')


def identity(pid):
    try:
        base=Path('/proc')/str(pid);fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],ppid=int(fields[1]),pgid=int(fields[2]),state=fields[0],
                    command=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except (OSError,ValueError,IndexError):return None


class InstanceClaim:
    def __init__(self,runtime):
        self.fd=os.open(runtime/'instance.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(self.fd);raise RuntimeError('Private instance is already claimed')
    def close(self):os.close(self.fd)


class OwnedProcesses:
    """Kernel ancestry establishes ownership; pidfds bind every signal to that identity."""
    def __init__(self,runtime,started_ns):
        self.runtime=runtime;self.started_ns=started_ns;self.known={};self.handles={}
        self.root=identity(os.getpid())
    def add(self,pid,parent=None):
        if pid in self.known:return
        parent=self.root if parent is None else parent
        try:fd=os.pidfd_open(pid)
        except ProcessLookupError:return
        current=identity(pid);owner=identity(parent['pid'])
        if (current is None or owner is None or owner['start_ticks']!=parent['start_ticks'] or
                current['ppid']!=parent['pid'] or select.select([fd],[],[],0)[0]):
            os.close(fd);return
        current['ownership_parent']=dict(pid=parent['pid'],start_ticks=parent['start_ticks'])
        self.known[pid]=current;self.handles[pid]=fd
    def scan(self):
        pending=[self.root,*self.known.values()];seen=set()
        while pending:
            parent=pending.pop();pid=parent['pid']
            if pid in seen:continue
            seen.add(pid);current=identity(pid)
            if current is None or current['start_ticks']!=parent['start_ticks']:continue
            children=[]
            try:
                for task in Path('/proc',str(pid),'task').iterdir():
                    try:children.extend(int(n) for n in (task/'children').read_text().split())
                    except OSError:pass
            except OSError:pass
            for child in children:
                self.add(child,parent)
                if child in self.known:pending.append(self.known[child])
    def remaining(self):
        remaining=[]
        for pid,old in self.known.items():
            now=identity(pid)
            if now is None or now['start_ticks']!=old['start_ticks']:continue
            if now['state']=='Z' and now['ppid']==os.getpid():
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
                now=identity(pid)
            if now and now['start_ticks']==old['start_ticks']:remaining.append(now)
        return remaining
    def send(self,pid,sig):
        old=self.known.get(pid);fd=self.handles.get(pid);now=identity(pid)
        if old is None or fd is None:raise RuntimeError('No proven child identity for cleanup')
        if now is None or now['start_ticks']!=old['start_ticks'] or select.select([fd],[],[],0)[0]:return
        try:signal.pidfd_send_signal(fd,sig)
        except ProcessLookupError:pass
    def stop(self):
        for sig,seconds in ((signal.SIGINT,12.),(signal.SIGTERM,8.),(signal.SIGKILL,3.)):
            self.scan()
            for row in self.remaining():
                if row['state']!='Z':self.send(row['pid'],sig)
            end=time.monotonic()+seconds
            while self.remaining() and time.monotonic()<end:self.scan();time.sleep(.1)
        self.scan()  # Include children reparented while their last owner exited.
        result=dict(tracked=list(self.known.values()),remaining=self.remaining())
        for fd in self.handles.values():os.close(fd)
        self.handles.clear()
        return result


def frozen(cut,repo=REPO):
    if not re.fullmatch('[0-9a-f]{40}',cut):raise ValueError('--cut must be a full commit hash')
    changed=subprocess.check_output(['git','diff','--name-only',cut,'--',*SCOPES],cwd=repo,text=True)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*SCOPES],cwd=repo,text=True)
    if changed or untracked:raise RuntimeError('Frozen smoke source differs: '+changed+untracked)


def plan(output,cut,instance=0):
    output=confined(output)
    if output.exists():raise ValueError('Smoke requires a fresh output directory')
    if instance!=0:raise ValueError('This smoke is assigned to instance 0/domain 90/GPU 0')
    wrapper=str(REPO/'linux-env/scripts/collect_sim.sh');stages=[dict(name='up',argv=['bash',wrapper,'up','0'])]
    stages.append(dict(name='clock-sanity',argv=['bash',wrapper,'clock-sanity','0','--output',str(output/'clock-sanity.json')]))
    for kind,seed in (('normal',330000),('regrasp',340000)):
        run=output/kind;base=['bash',wrapper,'run','0','--episodes','5','--seed',str(seed),'--diagnostic','--output',str(run)]
        if kind=='regrasp':base+=['--regrasp-only']
        stages.extend([dict(name=kind+'-first',argv=base+['--max-attempts','1']),
                       dict(name=kind+'-resume',argv=base+['--resume','--max-attempts','25']),
                       dict(name=kind+'-export',argv=['bash',wrapper,'export','0',str(run),'--repo-id','local/soarm101_'+kind+'_smoke','--report-dir',str(run/'report')])])
    return dict(cut=cut,output=str(output),instance=0,domain=90,gpu=0,bridge_port=8777,http_port=8778,
                stages=stages,diagnostic=True,review_clearance_claimed=False,pilot_started=False)


def restore_for_next_process(receipt):
    sys.path.insert(0,str(REPO/'omni-kit-mcp'))
    from kit_mcp.client import call
    code='import collection_runtime as c\nr=c.runtime()\nresult=r.restore('+repr(receipt['initial_physical'])+',arm=True)'
    result=call('run_python',dict(code=code),host='127.0.0.1',port=8777,timeout=30)
    if isinstance(result,dict) and 'result' in result:result=result['result']
    if not isinstance(result,dict) or not result.get('arm_restored') or not result.get('command_baseline'):
        raise RuntimeError('Between-process physical restore did not return its verified target baseline')
    return result


class StartupFailure(RuntimeError):
    def __init__(self, stage, code, log):
        self.diagnostic = dict(stage=stage, returncode=code, log=str(log),
            traceback=log.read_text(), attempts=0)
        super().__init__('Collector startup failed before an attempt receipt ('+stage+'):\n'+self.diagnostic['traceback'])


def first_receipt(run, stage, code, log):
    path=run/'attempt-0000/receipt.json'
    if not path.exists():raise StartupFailure(stage,code,log)
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cut',required=True);parser.add_argument('--output',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--go',action='store_true')
    args=parser.parse_args();recipe=plan(args.output,args.cut);frozen(args.cut)
    if args.plan:print(json.dumps(recipe,indent=2));return
    runtime=confined(CANONICAL_ROOT/'runtime/instance-0');output=Path(recipe['output'])
    claim=InstanceClaim(runtime)
    # Orphaned grandchildren remain our children even after their shell exits.
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot enable child subreaper')
    for name in ('isaacsim-soarm101.pid','control.pid'):
        path=runtime/name
        if path.exists() and identity(int(path.read_text())):raise RuntimeError('Private instance PID is already live; refusing startup: '+str(path))
    for port in (8777,8778):
        with socket.socket() as sock:
            sock.settimeout(.3)
            if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Private instance port is already live: '+str(port))
    if shutil.disk_usage('/').free<20*1024**3 or shutil.disk_usage(CANONICAL_ROOT).free<50*1024**3:raise RuntimeError('Smoke disk gate failed')
    output.mkdir(parents=True,exist_ok=False);(output/'plan.json').write_text(json.dumps(recipe,indent=2))
    env=dict(os.environ,ISAACSIM_GPU='0',SOARM_SIM_NICE='10',SOARM_PHYSICS_THREADS='4',
        SOARM_INSTANCE_ROOT=str(runtime),SOARM_COLLECTION_ROOT=str(CANONICAL_ROOT),HF_LEROBOT_HOME=str(CANONICAL_ROOT/'datasets'),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        SOARM_INSTANCE_CLAIM_FD=str(claim.fd),ISAACSIM_LOG=str(runtime/'isaac.log'))
    for name in ('CUDA_VISIBLE_DEVICES','ISAACSIM_EXTRA_ARGS','ROS_LOCALHOST_ONLY'):env.pop(name,None)
    ledger=OwnedProcesses(runtime,time.time_ns());stages=[];attempt_times=[];seen=set();complete=False;error=None;startup_failure=None
    (output/'runner-identity.json').write_text(json.dumps(identity(os.getpid()),indent=2))
    def stopped(signum,frame):raise KeyboardInterrupt('Smoke stop requested')
    signal.signal(signal.SIGTERM,stopped)
    def stage(item):
        frozen(args.cut);start=time.time_ns();last=start
        with (output/(item['name']+'.log')).open('w') as log:
            process=subprocess.Popen(item['argv'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(claim.fd,))
            ledger.add(process.pid)
            while process.poll() is None:
                ledger.scan()
                for path in sorted(output.glob('*/attempt-*/receipt.json')):
                    if str(path) not in seen:
                        receipt=json.loads(path.read_text());end=path.stat().st_mtime_ns
                        attempt_times.append(dict(path=str(path),stage=item['name'],accepted=receipt['accepted'],
                            error=receipt.get('error'),wall_interval_s=(end-last)/1e9));last=end;seen.add(str(path))
                time.sleep(.2)
            ledger.scan();row=dict(name=item['name'],pid=process.pid,returncode=process.returncode,wall_s=(time.time_ns()-start)/1e9)
            stages.append(row);(output/'stages.json').write_text(json.dumps(stages,indent=2));(output/'attempt-times.json').write_text(json.dumps(attempt_times,indent=2))
            print(json.dumps(row),flush=True);return process.returncode
    try:
        for item in recipe['stages']:
            code=stage(item);name=item['name']
            if name.endswith('-first'):
                kind=name.split('-')[0];run=output/kind;receipt=first_receipt(run,name,code,output/(name+'.log'))
                scene=json.loads((run/'first-reset-scene.json').read_text())
                if code not in (0,1) or not receipt['accepted'] or not receipt['timing']['passed'] or not scene['passed']:
                    raise RuntimeError(name+' did not pass; no next attempt is permitted')
                shutil.copy2(run/'first-reset-scene.json',output/(kind+'-first-reset-initial.json'))
                restored=restore_for_next_process(receipt);(output/(kind+'-between-process-restore.json')).write_text(json.dumps(restored,indent=2))
            elif code:raise RuntimeError(name+' failed; inspect its log and retained videos')
            if name.endswith('-resume'):
                run=output/name.split('-')[0];summary=json.loads((run/'summary.json').read_text())
                if not summary.get('complete') or summary.get('episodes')!=5:raise RuntimeError('Incomplete smoke receipt')
                if name.startswith('normal'):
                    receipt=json.loads((run/'attempt-0000/receipt.json').read_text())
                    (output/'before-regrasp-restore.json').write_text(json.dumps(restore_for_next_process(receipt),indent=2))
        for kind in ('normal','regrasp'):
            validation=json.loads((output/kind/'report/validation.json').read_text())
            if not validation['loaded'] or validation['fps']!=30 or validation['episodes']!=5 or not validation['timing_passed']:
                raise RuntimeError(kind+' export/load validation failed')
            if kind=='regrasp' and validation['recovery']['counts']['regrasp']!=5:raise RuntimeError('Dedicated recovery smoke lacks five recoveries')
        complete=True
    except BaseException as exc:
        error=str(exc) if isinstance(exc,StartupFailure) else repr(exc)
        startup_failure=exc.diagnostic if isinstance(exc,StartupFailure) else None
        raise
    finally:
        cleanup=ledger.stop();ports={}
        for port in (8777,8778):
            with socket.socket() as sock:sock.settimeout(.3);ports[str(port)]=sock.connect_ex(('127.0.0.1',port))==0
        result=dict(collection_complete=complete,error=error,startup_failure=startup_failure,cleanup=cleanup,ports_listening=ports,
                    passed=complete and not cleanup['remaining'] and not any(ports.values()),
                    visual_review_pending=True,independent_review_pending=True,pilot_started=False,cut=args.cut)
        (output/'smoke-receipt.json').write_text(json.dumps(result,indent=2))
        (output/'END').write_text('SMOKE_COLLECTION_FINISHED\n' if result['passed'] else 'SMOKE_STOPPED\n')
        if not cleanup['remaining'] and not any(ports.values()):print('Private instance and tracked children are down',flush=True)
        claim.close()
        if complete and not result['passed']:raise RuntimeError('Owned instance cleanup did not pass')


if __name__=='__main__':main()
