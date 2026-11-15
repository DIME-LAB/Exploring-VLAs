"""Filesystem handoff tests; no model load or GPU access."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import stage_checkpoints as w

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name).resolve()
        self.source=self.base/'source';self.source.mkdir();self.dest=self.base/'eval';self.dest.mkdir()
        self.source_patch=patch.object(w,'SOURCE_ROOT',str(self.source));self.source_patch.start()
        self.dest_patch=patch.object(w,'TARGET_ROOT',str(self.dest));self.dest_patch.start()
    def tearDown(self):
        self.dest_patch.stop();self.source_patch.stop();self.tmp.cleanup()
    def make(self,step):
        p=self.source/f'{step:06d}'/'pretrained_model';p.mkdir(parents=True)
        for name in w.REQUIRED:
            (p/name).write_bytes(b'weights' if name.endswith('.safetensors') else b'{}')
        state=p.parent/'training_state';state.mkdir();(state/'training_step.json').write_text(json.dumps({'step':step}))
        return p
    def marker(self,step):
        last=self.source/'last'
        if last.is_symlink():last.unlink()
        last.symlink_to(f'{step:06d}')
    def probe(self,step,hashes=False):
        return w.source_probe(dict(source_checkpoints=str(self.source),step=step,hashes=hashes))
    def test_existing_files_without_native_completion_marker_wait(self):
        self.make(10000);self.assertFalse(self.probe(10000)['ready'])
    def test_old_last_marker_does_not_qualify_new_save(self):
        self.make(10000);self.marker(10000);self.make(20000)
        self.assertFalse(self.probe(20000)['ready'])
    def test_native_completed_save_is_hashed(self):
        self.make(10000);self.marker(10000);r=self.probe(10000,True)
        self.assertTrue(r['ready']);self.assertEqual(len(r['files']['model.safetensors']['sha256']),64)
    def test_retired_optimizer_state_is_not_required_for_old_save(self):
        p=self.make(10000);self.make(20000);self.marker(20000)
        (p.parent/'training_state/training_step.json').unlink()
        self.assertTrue(self.probe(10000)['ready'])
    def test_missing_processor_state_is_refused(self):
        p=self.make(10000);self.marker(10000)
        (p/'policy_preprocessor.json').write_text(json.dumps({'state_file':'missing.safetensors'}))
        with self.assertRaises(ValueError):self.probe(10000)
    def test_source_model_symlink_is_refused(self):
        p=self.make(10000);self.marker(10000);(p/'model.safetensors').unlink();(p/'model.safetensors').symlink_to(self.base/'other-model')
        with self.assertRaises(ValueError):self.probe(10000)
    def test_existing_target_without_receipt_is_not_overwritten(self):
        final=self.dest/'010000';final.mkdir();(final/'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):w.prepare_target(dict(destination=str(self.dest),step=10000,bytes=0,transfer_id='test'))
        self.assertEqual((final/'sentinel').read_text(),'keep')
    def test_publish_atomically_preserves_receipt(self):
        request=dict(destination=str(self.dest),step=10000,bytes=0,transfer_id='test')
        prepared=w.prepare_target(request)
        out=w.publish_target(dict(request,directory=prepared['directory'],receipt={'step':10000}))
        self.assertTrue((Path(out['published'])/'handoff.json').is_file());self.assertFalse(Path(prepared['directory']).exists())
    def test_destination_outside_backup_root_refuses(self):
        with self.assertRaises(ValueError):w.prepare_target(dict(destination=str(self.base/'other'),step=10000,bytes=0,transfer_id='test'))

if __name__=='__main__':unittest.main(verbosity=2)
