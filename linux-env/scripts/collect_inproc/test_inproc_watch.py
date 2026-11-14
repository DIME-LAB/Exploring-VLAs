import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shared
from storage import CANONICAL_ROOT


@unittest.skipUnless(sys.platform=='linux' and CANONICAL_ROOT.exists(),'Linux canonical output root required')
class WatchTests(unittest.TestCase):
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
