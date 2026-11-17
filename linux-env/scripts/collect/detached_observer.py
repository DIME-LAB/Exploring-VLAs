"""Run a read-only notification observer independently of the launching tool session.

This launches only an observer, never the collector. A fresh runtime owns the log,
nonce-bound startup and final receipts. The observer keeps its own per-delivery
receipts; a launch receipt does not certify delivery or production completion.
"""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time,traceback,uuid
from pathlib import Path


def store(path,value):
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def birth(pid):
    if sys.platform=='linux':return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    # Informational only on macOS: this helper never authorizes a signal by PID.
    return subprocess.check_output(['ps','-p',str(pid),'-o','lstart='],text=True).strip()


def launch(script,runtime,arguments=()):
    script=Path(script).resolve(strict=True);runtime=Path(runtime).resolve()
    runtime.mkdir(parents=False,exist_ok=False)
    config=dict(script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                arguments=list(arguments),nonce=uuid.uuid4().hex)
    store(runtime/'launch.json',config)
    with (runtime/'observer.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'run',str(runtime)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    # No foreground tool-session handle remains responsible for its lifetime.
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        try:row=json.loads((runtime/'observer-running.json').read_text())
        except FileNotFoundError:row=None
        if row is not None:
            if row['nonce']!=config['nonce'] or row['pid']!=process.pid or row['pgid']!=process.pid or row['sid']!=process.pid:
                raise RuntimeError('Observer startup identity differs')
            if process.poll() is not None:
                raise RuntimeError('Observer exited during startup; inspect '+str(runtime/'observer-final.json'))
            threading.Thread(target=process.wait,daemon=True).start()
            return row
        if process.poll() is not None:raise RuntimeError('Observer failed at startup; inspect '+str(runtime/'observer.log'))
        time.sleep(.02)
    raise RuntimeError('Observer startup uncertified; inspect private runtime '+str(runtime))


def run(runtime):
    runtime=Path(runtime);config=json.loads((runtime/'launch.json').read_text());row=dict(config)
    row.update(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),ppid=os.getppid(),birth=birth(os.getpid()),started_wall_ns=time.time_ns())
    failure=None
    def interrupted(signum,frame):raise RuntimeError('Observer received signal '+str(signum))
    for sig in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    try:
        path=Path(config['script'])
        source=path.read_bytes()
        if hashlib.sha256(source).hexdigest()!=config['script_sha256']:raise RuntimeError('Observer source changed before startup')
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('notification_observer',path)
        module=importlib.util.module_from_spec(spec)
        exec(compile(source,str(path),'exec'),module.__dict__)
        sys.argv=[str(path),*config['arguments']]
        store(runtime/'observer-running.json',dict(row,state='running'))
        module.main()
    except BaseException as exc:
        failure=repr(exc);traceback.print_exc();raise
    finally:
        store(runtime/'observer-final.json',dict(row,state='finished' if failure is None else 'failed',
            passed=failure is None,error=failure,finished_wall_ns=time.time_ns()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    start=sub.add_parser('start');start.add_argument('--script',type=Path,required=True);start.add_argument('--runtime',type=Path,required=True);start.add_argument('arguments',nargs=argparse.REMAINDER)
    worker=sub.add_parser('run');worker.add_argument('runtime',type=Path)
    args=parser.parse_args()
    if args.operation=='run':run(args.runtime)
    else:
        arguments=args.arguments
        if arguments[:1]==['--']:arguments=arguments[1:]
        print(json.dumps(launch(args.script,args.runtime,arguments)))


if __name__=='__main__':main()
