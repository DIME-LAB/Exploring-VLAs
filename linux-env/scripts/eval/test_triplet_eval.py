"""CPU tests for triplet plans and the pre-transfer saved-processor gate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from triplet_bank import document, sha
from triplet_eval import plan, STEPS, paths, prepare
from checkpoint_contract import require_rule431_reference,require_transfer_processor_proof,checkpoint_files

ROOT=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/triplet360')


class Plans(unittest.TestCase):
    @unittest.skipUnless(Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/runs').exists(), 'owned dual integration output root required')
    def test_prepare_uses_real_native_out_contract(self):
        from triplet_eval import BASE
        self.assertTrue(paths(5000)['prep'].is_relative_to(BASE/'eval/runs'))
        with tempfile.TemporaryDirectory(dir=BASE/'eval/runs') as temporary:
            root=Path(temporary)
            p=dict(model=root/'pretrained_model',handoff=root/'handoff.json',proof=root/'proof.json',prep=root/'cpu')
            p['handoff'].write_text(json.dumps(dict(source_before=dict(model='/synthetic/unit-only',files={'model.safetensors':{'sha256':'a'*64}}))))
            calls=[]
            def run(argv,**kwargs):
                calls.append(argv)
                self.assertEqual(kwargs['env']['HF_HOME'],str(BASE/'eval/cache/huggingface'))
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                if '--check-only' in argv:
                    target=Path(argv[argv.index('--out')+1])
                    self.assertTrue(target.is_relative_to(BASE/'eval/runs'))
            with patch('triplet_eval.paths',return_value=p),patch('checkpoint_identity.validate_handoff_identity'),patch('checkpoint_contract.require_handoff',return_value={'files':{'triplet360-manifest.json':{}}}),patch('checkpoint_contract.require_processor_proof',return_value={'sha256':'b'*64}),patch('triplet_eval.load',return_value=([dict(seed=i) for i in range(60)],'triplet-heldout-20x3')),patch('triplet_eval.subprocess.run',side_effect=run):
                result=prepare(5000)
            self.assertEqual(len(calls),5)
            self.assertEqual(sum('--check-only' in argv for argv in calls),4)
            self.assertEqual(result['status'],'CPU_CHECKPOINT_READY_LIVE_QUALIFICATION_REQUIRED')
            self.assertEqual(len(result['selected_case_keys_by_offset']),4)
            # A new CPU revision may preflight only an unattempted ordered subset.
            calls.clear();p['prep']=root/'recovery'
            with patch('triplet_eval.paths',return_value=p),patch('checkpoint_identity.validate_handoff_identity'),patch('checkpoint_contract.require_handoff',return_value={'files':{'triplet360-manifest.json':{}}}),patch('checkpoint_contract.require_processor_proof',return_value={'sha256':'b'*64}),patch('triplet_eval.load',return_value=([dict(seed=i) for i in range(60)],'triplet-heldout-20x3')),patch('triplet_eval.subprocess.run',side_effect=run):
                recovered=prepare(5000,'v2',offset=0,episode_seeds=[2,3,4])
            self.assertEqual(len(calls),2)
            self.assertEqual(recovered['selected_case_keys_by_offset'],{'0':[2,3,4]})
            self.assertEqual(calls[-1][calls[-1].index('--episode-seeds')+1],'2,3,4')
            recovery_plan=plan('score',5000,0,'2026-11-20T00:00:00Z','a'*64,attempt=2,cpu_revision='v2',episode_seeds=[2,3,4])
            self.assertEqual(recovery_plan['argv'][recovery_plan['argv'].index('--episode-seeds')+1],'2,3,4')

    def test_qualification_attempt_independent_of_score_attempt(self):
        result=plan('score',5000,0,'2026-11-20T00:00:00Z','a'*64,attempt=3,qualification_attempt=2)
        self.assertEqual(result['argv'][result['argv'].index('--reset-qualification')+1],'/mnt/extreme4tb/aaugus11/exploring-vlas/eval/runs/t360-q-v2')

    def test_five_checkpoints_four_complete_chunks(self):
        for step in STEPS:
            for offset in (0,15,30,45):
                result=plan('score',step,offset,'2026-11-20T00:00:00Z','a'*64)
                self.assertEqual(result['gpu'],0)
                self.assertEqual(result['domain'],92)
                self.assertIn('--grasp-telemetry',result['argv'])
                self.assertIn('--paired-flow-repeat',result['argv'])
                self.assertEqual(result['argv'][result['argv'].index('--bank-count')+1],'15')
                self.assertIn('--matched-reset-arms',result['argv'])
        for offset in (1,14,60):
            with self.assertRaises(ValueError):plan('score',5000,offset,'2026-11-20T00:00:00Z','a'*64)

    def test_no_guessed_model_sha(self):
        with self.assertRaises(ValueError):plan('score',5000,0,'2026-11-20T00:00:00Z')


@unittest.skipUnless((ROOT/'contracts/fixture').exists(), 'sealed train360 processor fixture required')
class TransferGate(unittest.TestCase):
    def test_triplet_transfer_cannot_omit_pre_admission(self):
        from types import SimpleNamespace
        from stage_checkpoints import transfer
        with patch('stage_checkpoints.remote') as remote:
            with self.assertRaisesRegex(ValueError,'pre-transfer processor gate'):
                transfer(SimpleNamespace(completion_marker='triplet360-manifest.json'),{'step':5000})
            remote.assert_not_called()

    def test_published_triplet_handoff_cannot_omit_pre_admission(self):
        from checkpoint_contract import require_handoff
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            root=Path(temporary);model=root/'synthetic-unit-model';model.mkdir()
            (model/'model.safetensors').write_bytes(b'unit-only; not model weights')
            (model/'triplet360-manifest.json').write_text('{}')
            files=checkpoint_files(model)
            source=dict(ready=True,model=str(model),files=files)
            receipt=dict(published_model=str(model),stable_seconds=60,source_before=source,source_after=source,verified=dict(verified=True,files=files))
            path=root/'handoff.json';path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'pre-transfer actual-processor admission'):
                require_handoff(model,path,files['model.safetensors']['sha256'],published=True)

    @classmethod
    def setUpClass(cls):
        cls.fixture=ROOT/'contracts/fixture'
        cls.proof=document(ROOT/'proofs/training-processor-fixture.json')
        cls.files=checkpoint_files(cls.fixture)
        cls.docs={name:document(cls.fixture/name) for name in ('policy_preprocessor.json','policy_postprocessor.json','triplet360-manifest.json')}
        cls.snapshot=dict(model=cls.proof['checkpoint'],files=cls.files,processor_documents=cls.docs,
            completion=dict(artifact_marker=dict(name='triplet360-manifest.json')))

    def test_real_fixture_lineage_matches_train360_and_refuses_rule431_reference(self):
        self.assertFalse(self.proof['actual_training_checkpoint'])
        self.assertTrue(self.proof['fixture_only'])
        require_rule431_reference(self.fixture,self.proof,self.files,ROOT/'contracts/input-contract.json',family='triplet360')
        with self.assertRaises(ValueError):
            require_rule431_reference(self.fixture,self.proof,self.files,ROOT/'contracts/input-contract.json',family='rule431')

    def test_fixture_never_admitted_as_trained_save(self):
        with self.assertRaisesRegex(ValueError,'Actual saved processor proof'):
            require_transfer_processor_proof(self.snapshot,ROOT/'proofs/training-processor-fixture.json',ROOT/'contracts/input-contract.json','triplet360')

    def test_transfer_gate_synthetic_actual_flags_and_negative_post_stats(self):
        # This unit fixture changes flags only to exercise validation logic.
        # It never publishes a model or claims an actual training checkpoint.
        proof=copy.deepcopy(self.proof);proof.update(actual_training_checkpoint=True,fixture_only=False)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            path=Path(temporary)/'synthetic-unit-proof.json'
            path.write_text(json.dumps(proof))
            require_transfer_processor_proof(self.snapshot,path,ROOT/'contracts/input-contract.json','triplet360')
            proof['saved_normalizer_parity']['post_action']['mean'][0]+=1
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'normalizer values differ'):
                require_transfer_processor_proof(self.snapshot,path,ROOT/'contracts/input-contract.json','triplet360')


class RetainedSaveCompletion(unittest.TestCase):
    def evidence(self,last='010000',present=False,marker='triplet360-manifest.json',digest='a'*64):
        return dict(target=last,training_step_present=present,artifact_marker=dict(name=marker,sha256=digest))

    def test_retained_triplet_survives_native_optimizer_retention(self):
        from checkpoint_contract import require_save_completion
        result=require_save_completion('/unit-only/checkpoints/005000/pretrained_model',{'triplet360-manifest.json':{'sha256':'a'*64}},self.evidence())
        self.assertEqual(result['native_last'],10000)
        self.assertIn('pruned',result['basis'])

    def test_latest_missing_marker_still_refused(self):
        from checkpoint_contract import require_save_completion
        for last in ('000000','005000'):
            with self.assertRaises(ValueError):require_save_completion('/unit-only/checkpoints/005000/pretrained_model',{'triplet360-manifest.json':{'sha256':'a'*64}},self.evidence(last=last))

    def test_historical_artifact_sha_and_family_required(self):
        from checkpoint_contract import require_save_completion
        for name,digest in (('triplet360-manifest.json','b'*64),('rule431-manifest.json','a'*64)):
            with self.assertRaises(ValueError):require_save_completion('/unit-only/checkpoints/005000/pretrained_model',{name:{'sha256':'a'*64}},self.evidence(marker=name,digest=digest))

    def test_existing_families_keep_optimizer_marker_requirement(self):
        from checkpoint_contract import require_save_completion
        files={'rule431-manifest.json':{'sha256':'a'*64}}
        with self.assertRaises(ValueError):require_save_completion('/unit-only/checkpoints/005000/pretrained_model',files,self.evidence(marker='rule431-manifest.json'))
        self.assertEqual(require_save_completion('/unit-only/checkpoints/005000/pretrained_model',files,self.evidence(present=True))['step'],5000)


if __name__=='__main__':unittest.main()
