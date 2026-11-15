import json,tempfile,unittest
from pathlib import Path
from compare_scored_conditions import CONTRACT,compare

class ComparisonTests(unittest.TestCase):
    def fixture(self,root,rtc,missing_close=False,invalid=False):
        root.mkdir();(root/'behaviour').mkdir()
        contract={k:'fixed' for k in CONTRACT};contract.update(rtc_enabled=rtc,pretrained='/model/020000/pretrained_model')
        (root/'summary.json').write_text(json.dumps(dict(contract=contract,overall={})))
        rows=[];physical=[]
        for seed in range(900000,900030):
            rows.append(dict(seed=seed,color='red',task='same',valid=not(invalid and seed==900000),reported_outcome='no-grasp',success=False,empty_ticks_pct=0,effective_action_fps=30))
            physical.append(dict(seed=seed,closing_distance_m=None if missing_close and seed==900001 else .05,min_tcp_target_m=.04,tcp_dwell_pct=80,ever_closed_ge2deg=not(missing_close and seed==900001)))
        (root/'episodes.json').write_text(json.dumps(rows));(root/'behaviour/episodes.json').write_text(json.dumps(physical))
    def test_missing_closure_and_invalid_have_separate_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            l,r=Path(d)/'left',Path(d)/'right';self.fixture(l,True);self.fixture(r,False,True,True)
            pairs,s=compare(l,r)
            self.assertEqual(len(pairs),30);self.assertEqual(s['valid_both_n'],29)
            self.assertEqual(s['closing_distance_m_paired_difference']['n'],28)
            self.assertEqual(s['conditions'][1]['closed_n'],29)
            self.assertIsNone(pairs[1]['closing_distance_m_difference'])
    def test_bank_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            l,r=Path(d)/'left',Path(d)/'right';self.fixture(l,True);self.fixture(r,False)
            p=r/'summary.json';s=json.loads(p.read_text());s['contract']['scene_bank_sha256']='different';p.write_text(json.dumps(s))
            with self.assertRaisesRegex(AssertionError,'scene_bank_sha256'):compare(l,r)
    def test_missing_seed_refused(self):
        with tempfile.TemporaryDirectory() as d:
            l,r=Path(d)/'left',Path(d)/'right';self.fixture(l,True);self.fixture(r,False)
            p=r/'episodes.json';rows=json.loads(p.read_text());p.write_text(json.dumps(rows[:-1]))
            with self.assertRaises(AssertionError):compare(l,r)
    def test_changing_model_and_rtc_together_refused(self):
        with tempfile.TemporaryDirectory() as d:
            l,r=Path(d)/'left',Path(d)/'right';self.fixture(l,True);self.fixture(r,False)
            p=r/'summary.json';s=json.loads(p.read_text());s['contract']['pretrained']='/model/030000/pretrained_model';p.write_text(json.dumps(s))
            with self.assertRaisesRegex(AssertionError,'exactly one'):compare(l,r)
if __name__=='__main__':unittest.main()
