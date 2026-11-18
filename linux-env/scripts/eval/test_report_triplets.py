import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from report_rule431 import audit
from report_triplets import aggregate,paired
from triplet_bank import load,document,SUITE,RESET

ROOT=Path('/mnt/a4500-backup/exploring-vlas/eval/triplet360')
BANK=ROOT/'held20-v2'
SHA='0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d'


@unittest.skipUnless(BANK.exists(), 'actual sealed dual triplet bank required')
class Reports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,_=load(BANK,SHA)
        sys.path.insert(0,'/mnt/a4500-backup/exploring-vlas/eval-native-triplet360/linux-env/scripts/collect')

    def window(self,path,offset,count,invalid=False,success=False,started='2026-11-18T00:00:00Z',selected=None):
        from rule_target import select_target
        path.mkdir()
        ledger=dict(suite=SUITE,scored=True,stopped_utc='2026-11-18T00:00:00Z',cleanup_exit_code=0,status='policy-finished',
            bank=str(BANK),bank_sha256=SHA,selected_seeds=selected if selected is not None else [r['seed'] for r in self.rows[offset:offset+15]],readiness_budget_s=180,paired_flow_repeat=0,started_utc=started,bank_offset=offset,fixed_chunk_count=15)
        (path/'window-ledger.json').write_text(json.dumps(ledger))
        chosen=[r for r in self.rows[offset:offset+15] if selected is None or r['seed'] in selected][:count]
        for i,b in enumerate(chosen):
            out=path/'default'/f'ep{i:02d}';out.mkdir(parents=True)
            proof=select_target(document(b['source_receipt'])['initial_physical'],b['color'])
            row=dict(seed=b['seed'],color=b['color'],valid=not(invalid and i==0),weights_sha256='a'*64,bank_sha256=SHA,
                reset_protocol=RESET,readiness_budget_s=180,paired_flow_repeat=0,requested_colour_success=success,
                requested_colour_grasp=success,rule_target_success=success,rule_target_grasp=success,wrong_color_lift=False,wrong_same_colour_target_grasp=False,
                rule_target_selection=proof,client={'flow_noise':__import__('flow_noise').episode_identity(b['layout_seed'],0)},rule_closure_confirmed_geometry={'xy_mm':12.,'yaw_deg':4.,'dz_mm':3.,'phase':'CONFIRMED'})
            (out/'result.json').write_text(json.dumps(row))
        return path

    def test_aborted_partial_window_not_reported_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temporary);window=self.window(root/'window',0,2,invalid=True)
            summary,rows=audit(window,root/'report')
            self.assertEqual(summary['coverage']['status'],'partial')
            self.assertEqual(len(summary['coverage']['missing']),13)
            self.assertEqual(summary['coverage']['invalid'],1)
            self.assertEqual(rows[0]['case_id'],self.rows[0]['case_id'])
            ledger=document(window/'window-ledger.json');ledger['scored']=False
            (window/'window-ledger.json').write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError,'scored=True'):audit(window,root/'unscored-report')

    def test_aggregate_full_and_partial_denominators_and_invalids(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary,patch('native_bank.load_bank',return_value=(self.rows,SUITE)):
            root=Path(temporary)
            windows=[self.window(root/f'w{i}',i*15,15,invalid=i==0) for i in range(4)]
            full=aggregate(windows,root/'full')
            self.assertEqual(full['coverage']['status'],'complete')
            self.assertEqual(full['coverage']['invalid_attempts'],1)
            self.assertEqual(full['metrics']['rule_target_success']['denominator'],60)
            self.assertEqual([full['per_colour'][c]['attempted'] for c in ('red','green','blue')],[20]*3)
            partial=aggregate(windows[:1],root/'partial')
            self.assertEqual(partial['coverage']['status'],'partial')
            self.assertEqual(partial['metrics']['rule_target_success']['denominator'],15)
            self.assertEqual(len(partial['coverage']['missing_cases']),45)
            with self.assertRaisesRegex(ValueError,'Duplicate window'):aggregate([windows[0],windows[0]],root/'duplicate')

    def test_later_success_never_replaces_first_failure_or_invalid(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary,patch('native_bank.load_bank',return_value=(self.rows,SUITE)):
            root=Path(temporary)
            first=self.window(root/'first',0,15,invalid=True,success=False)
            later=self.window(root/'later',0,15,success=True,started='2026-11-18T01:00:00Z')
            report=aggregate([later,first],root/'report')
            self.assertEqual(report['metrics']['rule_target_success']['count'],0)
            self.assertEqual(report['coverage']['all_attempts'],30)
            self.assertEqual(report['coverage']['primary_attempts'],15)
            self.assertEqual(report['coverage']['invalid_attempts'],1)
            self.assertFalse(report['primary_rows'][0]['valid'])

    def test_resume_only_unattempted_subset_completes_original_cohort(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary,patch('native_bank.load_bank',return_value=(self.rows,SUITE)):
            root=Path(temporary)
            first=self.window(root/'interrupted',0,2,invalid=True)
            remaining=[r['seed'] for r in self.rows[2:15]]
            recovery=self.window(root/'resume',0,13,success=True,started='2026-11-18T01:00:00Z',selected=remaining)
            others=[self.window(root/f'w{i}',i*15,15) for i in (1,2,3)]
            report=aggregate([recovery,*others,first],root/'report')
            self.assertEqual(report['coverage']['status'],'complete')
            self.assertEqual(report['coverage']['primary_attempts'],60)
            self.assertEqual(report['coverage']['all_attempts'],60)
            self.assertEqual(report['metrics']['rule_target_success']['count'],13)
            self.assertEqual(report['coverage']['invalid_attempts'],1)

    def test_paired_bootstrap_resamples_whole_rgb_layout(self):
        rows=[dict(layout_seed=i,color=c,valid=True,rule_target_success=False,paired_flow_repeat=0) for i in range(2) for c in ('red','green','blue')]
        left=dict(bank_sha256=SHA,primary_rows=rows);right=copy.deepcopy(left)
        for row in right['primary_rows'][:3]:row['rule_target_success']=True
        result=paired(left,right,draws=500)
        self.assertEqual(result['shared_cases'],6)
        self.assertEqual(result['differences_right_minus_left']['rule_target_success']['mean'],.5)
        self.assertEqual(result['differences_right_minus_left']['rule_target_success']['layout_cluster_percentile95'],[0.,1.])
        right['primary_rows'][0]['paired_flow_repeat']=1
        with self.assertRaisesRegex(ValueError,'inference repeats'):paired(left,right,draws=500)


if __name__=='__main__':unittest.main()
