import copy,unittest
from checkpoint_identity import checkpoint_label,validate_handoff_identity

class IdentityTests(unittest.TestCase):
 def setUp(self):
  self.model='/target/unfreeze30k/002500/pretrained_model';self.source='/source/job/unfreeze-10k/training/checkpoints';path=self.source+'/002500/pretrained_model';files={'model.safetensors':dict(size=42,sha256='fixed',mtime_ns=9)}
  self.receipt=dict(step=2500,published_model=self.model,source=path,source_before=dict(ready=True,step=2500,model=path,files=files),source_after=dict(ready=True,step=2500,model=path,files=copy.deepcopy(files)),verified=dict(verified=True,files=copy.deepcopy(files)))
 def validate(self):validate_handoff_identity(self.receipt,self.model,2500,self.source)
 def test_identity_accepts_published_verified_same_step(self):self.validate()
 def test_families_have_distinct_labels(self):
  self.assertEqual(checkpoint_label('/primary/010000/pretrained_model'),'10k');self.assertEqual(checkpoint_label('/target/unfreeze30k/010000/pretrained_model'),'unfreeze30k+10000')
 def test_wrong_step_rejected(self):
  self.receipt['step']=5000
  with self.assertRaisesRegex(ValueError,'step'):self.validate()
 def test_wrong_source_job_rejected(self):
  self.receipt['source']='/source/another-job/unfreeze-10k/training/checkpoints/002500/pretrained_model'
  with self.assertRaisesRegex(ValueError,'source job'):self.validate()
 def test_wrong_published_target_rejected(self):
  self.receipt['published_model']='/target/unfreeze30k/005000/pretrained_model'
  with self.assertRaisesRegex(ValueError,'destination'):self.validate()
 def test_source_change_rejected(self):
  self.receipt['source_after']['files']['model.safetensors']['sha256']='changed'
  with self.assertRaisesRegex(ValueError,'changed'):self.validate()
 def test_unverified_target_rejected(self):
  self.receipt['verified']['verified']=False
  with self.assertRaisesRegex(ValueError,'not verified'):self.validate()
 def test_wrong_target_content_rejected(self):
  self.receipt['verified']['files']['model.safetensors']['sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'bytes differ'):self.validate()
if __name__=='__main__':unittest.main()
