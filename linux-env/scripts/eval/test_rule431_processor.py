"""Reference-guard regressions from a real saved processor fixture, not weights."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from checkpoint_contract import require_rule431_reference,require_processor_proof

class RuleProcessorTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parent
        self.fixture=json.loads((root/'fixtures/rule431-processor-reference.json').read_text())
        self.assertTrue(self.fixture['fixture_only'])
        self.reference=root.parent/'contracts/input-contract.json'
        # In-repo tests use the delivered, SHA-bound reference files beside the fixture.
        if not self.reference.exists():self.reference=root/'fixtures/rule431-contract/input-contract.json'
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        for name,text in self.fixture['serialized_files'].items():(self.directory/name).write_text(text)
        self.proof=copy.deepcopy(self.fixture['proof'])
        self.files={name:dict(sha256=sha) for field in ['processor_config_files_sha256','processor_state_files_sha256'] for name,sha in self.proof[field].items()}
        self.files['rule431-manifest.json']=dict(sha256=hashlib.sha256((self.directory/'rule431-manifest.json').read_bytes()).hexdigest())
    def test_actual_fixture_metadata_matches_new_subset(self):
        result=require_rule431_reference(self.directory,self.proof,self.files,self.reference)
        self.assertEqual(result['normalization_stats_sha256'],self.proof['normalization_stats_sha256'])
    def test_wrong_post_action_stats_or_old_reference_refuses(self):
        bad=copy.deepcopy(self.proof);bad['saved_normalizer_parity']['post_action']['mean'][0]+=1
        with self.assertRaisesRegex(ValueError,'normalizer values'):require_rule431_reference(self.directory,bad,self.files,self.reference)
        bad=copy.deepcopy(self.proof);bad['reference_processor_proof_sha256']='0'*64
        with self.assertRaises(ValueError):require_rule431_reference(self.directory,bad,self.files,self.reference)
        bad=copy.deepcopy(self.proof);bad['processor_state_files_sha256'][next(iter(bad['processor_state_files_sha256']))]='0'*64
        with self.assertRaises(ValueError):require_rule431_reference(self.directory,bad,self.files,self.reference)
    def test_fixture_remains_inadmissible_as_actual_checkpoint(self):
        p=self.directory/'proof.json';p.write_text(json.dumps(self.proof))
        with self.assertRaisesRegex(ValueError,'Actual saved training processor proof required'):
            require_processor_proof(self.directory,p,self.files,reference_path=self.reference)

if __name__=='__main__':unittest.main()
