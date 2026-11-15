import hashlib,json,tempfile,unittest
from pathlib import Path
from checkpoint_contract import checkpoint_files,require_handoff
class Tests(unittest.TestCase):
    def test_processor_tamper_is_refused_even_when_weights_are_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'model';model.mkdir()
            (model/'model.safetensors').write_bytes(b'weights');(model/'normalizer.safetensors').write_bytes(b'normalizer')
            inventory=checkpoint_files(model)
            receipt=root/'receipt.json';receipt.write_text(json.dumps(dict(source_before=dict(model=str(model),files=inventory),source_after=dict(files=inventory),verified=dict(verified=True))))
            sha=hashlib.sha256(b'weights').hexdigest();require_handoff(model,receipt,sha)
            (model/'normalizer.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'files differ'):require_handoff(model,receipt,sha)
    def test_symlinked_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'model';model.mkdir();(root/'outside').write_text('module')
            (model/'module.py').symlink_to(root/'outside')
            with self.assertRaisesRegex(ValueError,'symlink'):checkpoint_files(model)
if __name__=='__main__':unittest.main()
