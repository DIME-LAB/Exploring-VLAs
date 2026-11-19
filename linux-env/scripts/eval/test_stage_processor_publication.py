"""CPU-only tests for opt-in atomic publication of already-admitted proof bytes."""
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
SOURCE=Path(__file__).with_name('stage_checkpoints.py')
if not SOURCE.exists():SOURCE=Path('/Users/aldrininbaraj/Documents/Projects/untitled folder/agents/ros2/coursework-eee598/roadmap-2026-11/phase3/3.3-eval/triplet360/code/stage_checkpoints.py')
spec=importlib.util.spec_from_file_location('stage_checkpoints',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Publication(unittest.TestCase):
 def request(self,folder):
  text=json.dumps({'fixture_only':True,'actual_training_checkpoint':False})+'\n'
  return dict(root=str(folder/'proofs'),step=5000,text=text,sha256=hashlib.sha256(text.encode()).hexdigest(),transfer_id='a'*32)
 def test_atomic_and_immutable(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t).resolve();r=self.request(root)
   with patch.object(m,'TARGET_ROOT',str(root)),patch.object(shutil,'disk_usage',return_value=types.SimpleNamespace(free=200*1024**3)):
    result=m.publish_processor_proof(r);self.assertEqual(Path(result['path']).read_bytes(),r['text'].encode())
    self.assertEqual(list(Path(r['root']).rglob('*.tmp')),[]);self.assertEqual(result,m.publish_processor_proof(r))
    r['text']='{}';r['sha256']=hashlib.sha256(b'{}').hexdigest()
    with self.assertRaises(FileExistsError):m.publish_processor_proof(r)
 def test_tamper_outside_and_symlink_refused(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t).resolve()
   with patch.object(m,'TARGET_ROOT',str(root)):
    r=self.request(root);r['sha256']='b'*64
    with self.assertRaises(ValueError):m.publish_processor_proof(r)
    r=self.request(root);r['root']='/tmp/unowned-proof-root'
    with self.assertRaises(ValueError):m.publish_processor_proof(r)
    (root/'actual').mkdir();(root/'proofs').symlink_to(root/'actual',target_is_directory=True);r=self.request(root)
    with self.assertRaises(ValueError):m.publish_processor_proof(r)
 def test_floor_refused(self):
  with tempfile.TemporaryDirectory() as t,patch.object(shutil,'disk_usage',return_value=types.SimpleNamespace(free=99*1024**3)),patch.object(m,'TARGET_ROOT',str(Path(t).resolve())):
   with self.assertRaises(RuntimeError):m.publish_processor_proof(self.request(Path(t).resolve()))
 def test_publish_precedes_handoff_transfer_and_default_unchanged(self):
  class StopBeforeTransfer(Exception):pass
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r=self.request(root);proof=root/'proof.json';proof.write_text(r['text'])
   expected=dict(ready=True,step=5000,files={},model='fixture-only')
   admission=dict(path=str(proof),sha256=r['sha256']);fake=types.SimpleNamespace(require_transfer_processor_proof=lambda *a:admission)
   for enabled in (False,True):
    calls=[]
    def remote(host,function,request):
     calls.append(function)
     if function=='source_probe':return expected
     if function=='prepare_target':raise StopBeforeTransfer()
     return {'published':True}
    args=argparse.Namespace(completion_marker='triplet360-manifest.json',processor_proof_directory=root,processor_family='triplet360',processor_reference=root/'reference.json',published_processor_proof_root=root/'proofs' if enabled else None,source_host='fixture',source_checkpoints='fixture',target_host='fixture',destination='fixture')
    with patch.dict(sys.modules,{'checkpoint_contract':fake}),patch.object(m,'remote',remote):
     with self.assertRaises(StopBeforeTransfer):m.transfer(args,dict(step=5000,files={}))
    self.assertEqual(calls,['source_probe']+(['publish_processor_proof'] if enabled else [])+['prepare_target'])

if __name__=='__main__':unittest.main()
