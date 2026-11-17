"""Real standalone owner bank loading must precede the deliberate past-deadline refusal."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path
class Tests(unittest.TestCase):
 def test_exact_owner_entrypoint_loads_rule_bank_in_clean_interpreter(self):
  bank=Path('/mnt/a4500-backup/exploring-vlas/eval/rule-seen15-v1')
  if not bank.is_dir():self.skipTest('actual sealed rule bank is dual-only')
  repo=Path(__file__).resolve().parents[3]
  if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!='bf7336d93d5ba054c5edc66cea685725eb30eb40':self.skipTest('exact owner import regression requires the private bf7336d collector runtime')
  with tempfile.TemporaryDirectory(dir='/mnt/a4500-backup/exploring-vlas/eval/tmp') as d:
   cmd=[sys.executable,'-E','-B',str(Path(__file__).with_name('run_native_replay_window.py')),'--matched-reset-qualification','--out','/mnt/a4500-backup/exploring-vlas/eval/runs/cpu-rule-import-must-not-exist','--bank',str(bank),'--bank-sha256','b4bc43f02af3d727bb4f55febc76da0cf7a61813027e7ad042de2b15e2d2383d','--reset-collector-root','/mnt/a4500-backup/exploring-vlas/eval-bank-bf7336d','--stop-by-utc','2020-01-01T00:00:00Z']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',SOARM_COLLECTION_ROOT='/mnt/a4500-backup/exploring-vlas');env.pop('PYTHONPATH',None)
   result=subprocess.run(cmd,env=env,cwd=d,text=True,capture_output=True)
   self.assertNotEqual(result.returncode,0)
   self.assertIn('assert deadline>time.time()+600',result.stderr,result.stderr)
   self.assertNotIn('ModuleNotFoundError',result.stderr);self.assertNotIn('ImportError',result.stderr)
   self.assertFalse(Path('/mnt/a4500-backup/exploring-vlas/eval/runs/cpu-rule-import-must-not-exist').exists())
if __name__=='__main__':unittest.main()
