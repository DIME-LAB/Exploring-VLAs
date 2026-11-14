import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import shared
from storage import CANONICAL_ROOT


@unittest.skipUnless(sys.platform=='linux' and CANONICAL_ROOT.exists(),'Linux canonical output root required')
class WatchTests(unittest.TestCase):
    def test_boot_configuration_uses_persistent_driver_cache(self):
        environment=dict(SOARM_INSTANCE_ROOT='/runtime/private',SOARM_RENDERER_GPU_INDEX='0',
            SOARM_CURRENT_KIT_LOG='/runtime/private/kit.log',SOARM_SHADER_CACHE_DIR='/cache/card/shaders')
        with patch.dict(os.environ,environment),patch.object(sys,'argv',['probe.py','--already-parsed']):
            config=shared.app_config()
            self.assertIn('--/rtx/shaderDb/driverShaderCachePath=/cache/card/vulkan-driver',config['extra_args'])
            for key in ('/persistent/omni/replicator/captureOnPlay','/omni/replicator/captureOnPlay'):
                self.assertIn('--'+key+'=true',config['extra_args'])
            self.assertEqual(sys.argv,['probe.py','--portable-root','/runtime/private/portable'])
            self.assertTrue(config['fast_shutdown']);self.assertFalse(config['multi_gpu'])

    def test_failure_readiness_without_a_summary_is_retained(self):
        parent=CANONICAL_ROOT/'inproc/offline-tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root=Path(directory)
            shared.publish_exit_ready(root,dict(invocation='failed-boot'),False)
            ready=json.loads((root/'worker-exit-ready.json').read_text())
            self.assertFalse(ready['passed']);self.assertIsNone(ready['summary_sha256'])
            with self.assertRaises(RuntimeError):
                shared.publish_exit_ready(root,dict(invocation='missing-summary'),True)

    def test_owner_finalizes_after_a_ready_worker_exits(self):
        parent=CANONICAL_ROOT/'inproc/offline-tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            runtime=Path(directory);output=runtime/'output'
            code='''import sys,os,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import shared
from storage import atomic_json
from collection_status import begin
root=Path(sys.argv[-1]);root.mkdir()
c=begin(root)
atomic_json(root/'summary.json',dict(complete=True,end_marker='COLLECTION_FINISHED',producer_invocation=c['invocation']))
atomic_json(root/'worker-exit-ready.json',dict(version=1,passed=True,producer_pid=os.getpid(),invocation=c['invocation'],summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()))
'''
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('watch.py')),'--',
                sys.executable,'-c',code,str(Path(__file__).parent),'--output',str(output)],
                env=dict(os.environ,SOARM_INSTANCE_ROOT=str(runtime)),capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads((runtime/'inproc-watchdog.json').read_text())
            certificate=json.loads((output/'collector-finalization.json').read_text())
            self.assertTrue(receipt['passed']);self.assertFalse(receipt['cleanup']['remaining'])
            self.assertTrue(certificate['passed']);self.assertTrue(certificate['completed'])

    def invoke(self,code):
        parent=CANONICAL_ROOT/'inproc/offline-tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            runtime=Path(directory)
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('watch.py')),'--',
                sys.executable,'-c',code,'--output',str(runtime/'output')],
                env=dict(os.environ,SOARM_INSTANCE_ROOT=str(runtime)),capture_output=True,text=True,timeout=15)
            receipt=json.loads((runtime/'inproc-watchdog.json').read_text())
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(receipt['passed'])
            self.assertFalse(receipt['cleanup']['remaining'])
            return receipt
    def test_zero_exit_without_final_receipt_is_refused(self):
        self.assertEqual(self.invoke('pass')['failure']['reason'],'finalization_failure')
    def test_child_failure_and_owned_cleanup(self):
        self.assertEqual(self.invoke('raise SystemExit(7)')['worker_returncode'],7)


if __name__=='__main__':unittest.main()
