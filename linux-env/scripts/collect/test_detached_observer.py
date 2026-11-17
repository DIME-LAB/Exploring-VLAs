"""An observer must survive its launching tool/shell and then deliver a late event."""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import detached_observer


@unittest.skipUnless(os.name=='posix','POSIX sessions required')
class DetachedObserverTests(unittest.TestCase):
    def wait(self,predicate):
        end=time.monotonic()+10
        while time.monotonic()<end:
            result=predicate()
            if result:return result
            time.sleep(.02)
        self.fail('Observer condition exceeded deadline')

    def fixture(self,root,fail=False):
        script=root/'observer.py'
        script.write_text("""import json,sys,time
from pathlib import Path
def main():
 root=Path(sys.argv[1]);deadline=time.monotonic()+8
 while not (root/'STOP-source.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('No event before fixture deadline')
  time.sleep(.02)
 source=json.loads((root/'STOP-source.json').read_text())
 (root/'delivered.json').write_text(json.dumps(source))
"""+(' raise RuntimeError("original observer failure")\n' if fail else ''))
        return script

    def test_launch_group_hup_int_and_exit_leave_detached_observer_delivering_late_stop(self):
        for signum in (signal.SIGHUP,signal.SIGINT,None):
            with self.subTest(signum=signum),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);script=self.fixture(root);runtime=root/'private-runtime'
                code="""import json,sys,time
from pathlib import Path
from detached_observer import launch
root=Path(sys.argv[1]);row=launch(root/'observer.py',root/'private-runtime',[str(root)])
(root/'launcher-ready.json').write_text(json.dumps(row))
time.sleep(8)
"""
                launcher=subprocess.Popen([sys.executable,'-B','-c',code,str(root)],
                    env=dict(os.environ,PYTHONPATH=str(Path(detached_observer.__file__).parent)),
                    start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                try:
                    self.wait(lambda:(root/'launcher-ready.json').exists())
                    row=json.loads((root/'launcher-ready.json').read_text())
                    self.assertEqual(row['pid'],row['sid']);self.assertEqual(row['pid'],row['pgid'])
                    self.assertNotEqual(row['sid'],launcher.pid)
                    if signum is None:launcher.terminate()
                    else:os.killpg(launcher.pid,signum)
                    launcher.wait(timeout=5)
                    source=dict(kind='STOP',original_error='pidfd_open EINVAL',production_success=False)
                    (root/'STOP-source.json').write_text(json.dumps(source))
                    self.wait(lambda:(runtime/'observer-final.json').exists())
                    self.assertEqual(json.loads((root/'delivered.json').read_text()),source)
                    final=json.loads((runtime/'observer-final.json').read_text())
                    self.assertTrue(final['passed']);self.assertEqual(final['nonce'],row['nonce'])
                    self.assertEqual(final['script_sha256'],row['script_sha256'])
                finally:
                    if launcher.poll() is None:launcher.terminate();launcher.wait(timeout=5)
                    if not (root/'STOP-source.json').exists():(root/'STOP-source.json').write_text('{}')
                    self.wait(lambda:(runtime/'observer-final.json').exists())

    def test_failed_observer_persists_original_error_and_never_certifies_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'private-runtime';script=self.fixture(root,fail=True)
            row=detached_observer.launch(script,runtime,[str(root)])
            (root/'STOP-source.json').write_text('{"kind":"STOP"}')
            self.wait(lambda:(runtime/'observer-final.json').exists())
            final=json.loads((runtime/'observer-final.json').read_text())
            self.assertFalse(final['passed']);self.assertEqual(final['nonce'],row['nonce'])
            self.assertIn('original observer failure',final['error'])
            self.assertIn('original observer failure',(runtime/'observer.log').read_text())

    def test_existing_runtime_refuses_a_second_observer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);script=self.fixture(root)
            with self.assertRaises(FileExistsError):detached_observer.launch(script,root)


    def test_changed_observer_source_refuses_before_running_and_records_failure(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);script=self.fixture(root);runtime=root/'private-runtime';runtime.mkdir()
            config=dict(script=str(script),script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),arguments=[str(root)],nonce='test-changed-source')
            detached_observer.store(runtime/'launch.json',config)
            script.write_text("def main():\n raise AssertionError('Changed source ran')\n")
            worker=subprocess.run([sys.executable,'-B',detached_observer.__file__,'run',str(runtime)],capture_output=True,text=True)
            self.assertNotEqual(worker.returncode,0);self.assertFalse((runtime/'observer-running.json').exists())
            final=json.loads((runtime/'observer-final.json').read_text());self.assertFalse(final['passed'])
            self.assertIn('source changed before startup',final['error'])
