import hashlib,json,tempfile,unittest
from pathlib import Path
from checkpoint_contract import checkpoint_files,require_handoff,require_processor_proof
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
    def test_midpoint_or_fixture_proof_cannot_qualify_an_end_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'009102/pretrained_model';model.mkdir(parents=True)
            (model/'workspace_crop_step.py').write_text('module');files=checkpoint_files(model)
            proof=dict(passed=True,cpu_only=True,training_written_processor_reloaded=True,fixture_only=False,checkpoint=str(model),completed_last_marker=str(model.parent),module_sha256=files['workspace_crop_step.py']['sha256'],raw_policy_input_keys=['observation.state','observation.images.top','observation.images.wrist'],stable_file_sizes={'workspace_crop_step.py':6},pixel_state_token_and_resize_hashes={k:'0'*64 for k in ('observation.state','observation.images.top','observation.images.wrist','observation.language.tokens','top_after_native_resize')})
            path=root/'proof.json';path.write_text(json.dumps(proof));require_processor_proof(model,path,files)
            proof['fixture_only']=True;path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'Actual saved'):require_processor_proof(model,path,files)
            proof['fixture_only']=False;proof['checkpoint']=str(root/'004551/pretrained_model');path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'another checkpoint'):require_processor_proof(model,path,files)
if __name__=='__main__':unittest.main()
