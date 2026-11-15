"""Notifier ownership and acknowledgement survive the launcher's retirement."""
import json,os,shlex,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path


class NotificationTests(unittest.TestCase):
    def wait(self,read):
        end=time.monotonic()+10
        while time.monotonic()<end:
            value=read()
            if value:return value
            time.sleep(.03)
        self.fail('Notifier did not publish the expected receipt')

    def fixture(self,root,fail=False):
        source=root/'monitor.json';source.write_text(json.dumps(dict(wall_ns=1,
            slots={str(i):dict(selected_cohorts=0) for i in (3,4)},faults=[],stopping=False,final=None)))
        code='import os; from pathlib import Path; print(Path(os.environ["MONITOR_FIXTURE"]).read_text())'
        ssh=root/'ssh';ssh.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' -c '+shlex.quote(code)+'\n');ssh.chmod(0o755)
        ask=root/'ask.py';ask.write_text('import json,sys\nfrom pathlib import Path\na=sys.argv\np=Path(a[a.index("--outcome")+1]);p.write_text(json.dumps({"kind":"'+('unknown' if fail else 'accepted')+'"}))\nraise SystemExit('+str(int(fail))+')\n')
        env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],MONITOR_FIXTURE=str(source),PYTHONDONTWRITEBYTECODE='1')
        out=root/'out';script=Path(__file__).with_name('notify_collection.py')
        command=[sys.executable,str(script),'--spawn','--remote-root','/fake/receipts','--out',str(out),
            '--ask-script',str(ask),'--recipient','fake-session','--interval','.04']
        launched=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
        return source,out,json.loads(launched.stdout)

    def update(self,path,**changes):
        value=json.loads(path.read_text());value.update(changes)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)

    def test_detached_worker_delivers_fault_after_launcher_exit_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            source,out,ready=self.fixture(Path(d));pid=ready['pid']
            try:
                self.assertEqual(ready['pgid'],pid);self.assertNotEqual(ready['pgid'],os.getpgid(0))
                self.wait(lambda:json.loads((out/'heartbeat.json').read_text())['wall_ns']>ready['wall_ns'])
                self.update(source,wall_ns=2,faults=[dict(error='Synthetic request dimensions differ')],stopping=True)
                state=self.wait(lambda:json.loads((out/'state.json').read_text()) if (out/'state.json').exists() and json.loads((out/'state.json').read_text())['events'].get('stop',{}).get('kind')=='accepted' else None)
                time.sleep(.15);self.assertEqual(len(json.loads((out/'state.json').read_text())['events']),1)
                self.update(source,wall_ns=3,final=dict(passed=False,owned_down=True))
                done=self.wait(lambda:json.loads((out/'completed.json').read_text()) if (out/'completed.json').exists() else None)
                self.assertTrue(done['passed']);self.assertEqual(set(done['events']),{'stop','final'})
                self.assertTrue(all(e['kind']=='accepted' for e in done['events'].values()))
            finally:
                # The successful worker retires itself; only a failed fixture is signalled.
                if not (out/'completed.json').exists():
                    from notify_collection import birth
                    if birth(pid)==ready['start_ticks']:
                        fd=os.pidfd_open(pid)
                        try:
                            if birth(pid)==ready['start_ticks']:signal.pidfd_send_signal(fd,signal.SIGTERM)
                        finally:os.close(fd)

    def test_unknown_send_is_unresolved_never_acknowledged_or_retried(self):
        with tempfile.TemporaryDirectory() as d:
            source,out,ready=self.fixture(Path(d),fail=True)
            self.update(source,wall_ns=2,faults=[dict(error='fault')],stopping=True)
            error=self.wait(lambda:json.loads((out/'error.json').read_text()) if (out/'error.json').exists() else None)
            state=json.loads((out/'state.json').read_text())
            self.assertIn('not acknowledged',error['error']);self.assertEqual(state['events']['stop']['kind'],'unresolved')
            self.assertFalse((out/'completed.json').exists());self.assertEqual(len(list(out.glob('*-outcome.json'))),1)


if __name__=='__main__':unittest.main()
