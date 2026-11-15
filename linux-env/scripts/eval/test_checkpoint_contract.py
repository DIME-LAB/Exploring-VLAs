import hashlib,json,tempfile,unittest
from pathlib import Path
from checkpoint_contract import checkpoint_files,require_handoff,require_processor_proof
class Tests(unittest.TestCase):
    def test_robustness_proof_needs_complete_handoff_and_no_random_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'target';model.mkdir();(model/'workspace_crop_step.py').write_text('module')
            steps=[dict(registry_name=n,config={}) for n in ('rename_observations_processor','to_batch_processor','smolvla_new_line_processor','tokenizer_processor','device_processor','normalizer_processor')]
            steps.insert(2,{'class':'workspace_crop_step.WorkspaceTopCropStep','config':{'key':'observation.images.top','box_xyxy':[99,128,559,473],'input_hw':[480,640]}});steps[-1]['config']={'norm_map':{'VISUAL':'IDENTITY'}}
            (model/'policy_preprocessor.json').write_text(json.dumps({'steps':steps}));files=checkpoint_files(model)
            handoff=dict(published_model=str(model),source_before=dict(ready=True,model='/source/005000/pretrained_model',files=files,completion=dict(target='005000',training_step_present=True)),source_after=dict(ready=True,files=files),verified=dict(verified=True,files=files),stable_seconds=60);h=root/'handoff.json';h.write_text(json.dumps(handoff))
            proof=dict(passed=True,cpu_only=True,actual_training_checkpoint=True,fixture_only=False,checkpoint='/source/005000/pretrained_model',crop_module_sha256=files['workspace_crop_step.py']['sha256'],pixel_state_token_resize_hashes={k:'0'*64 for k in ('observation.images.top','observation.images.wrist','observation.language.tokens','observation.state','top_after_resize')},raw_image_shapes=[3,480,640],inference_order=['top_crop','normal_resize'],no_random_inference_augmentation=True);p=root/'proof.json';p.write_text(json.dumps(proof))
            require_processor_proof('/source/005000/pretrained_model',p,files,published_handoff=h)
            with self.assertRaisesRegex(ValueError,'completed published'):require_processor_proof('/source/005000/pretrained_model',p,files)
            proof['no_random_inference_augmentation']=False;p.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'image contract'):require_processor_proof('/source/005000/pretrained_model',p,files,published_handoff=h)
    def test_published_copy_requires_exact_target_and_verified_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);model=root/'published';model.mkdir();(model/'model.safetensors').write_bytes(b'weights');inventory=checkpoint_files(model);sha=inventory['model.safetensors']['sha256']
            receipt=dict(published_model=str(model),source_before=dict(ready=True,model='/source/model',files=inventory),source_after=dict(ready=True,files=inventory),stable_seconds=60,verified=dict(verified=True,files=inventory))
            path=root/'receipt.json';path.write_text(json.dumps(receipt));require_handoff(model,path,sha,published=True)
            with self.assertRaisesRegex(ValueError,'path differs'):require_handoff(model,path,sha)
            receipt['verified']['files']={};path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'verification differs'):require_handoff(model,path,sha,published=True)
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
