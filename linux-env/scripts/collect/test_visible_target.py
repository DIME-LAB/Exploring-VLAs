"""Visible selection, default parameter parity and sealed-set analysis boundaries."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from contract import parameters,visible_target,apply_target_rule,VISIBLE_TARGET_RULE
from analyze_target_rule import analyze,publish


def snapshot():
    return dict(base_world_pose=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]),
        bodies={f'{c}_2x3_{i}':dict(position=[.1+.1*i,.01*j,0.])
            for j,c in enumerate(('red','green','blue')) for i in range(2)})


class VisibleTargetTests(unittest.TestCase):
    def test_xy_distance_uses_fixed_base_frame_and_ignores_height(self):
        s=snapshot();s['base_world_pose']=dict(position=[1.,2.,3.],orientation=[math.sqrt(.5),0.,0.,math.sqrt(.5)])
        s['bodies']['red_2x3_0']['position']=[.8,2.1,103.]
        s['bodies']['red_2x3_1']['position']=[1.,2.3,3.]
        p=visible_target(s,'red',True)
        self.assertEqual(p['object'],'red_2x3_0');self.assertAlmostEqual(p['candidates'][0]['distance_xy_m'],math.hypot(.1,.2))
        self.assertAlmostEqual(p['candidates'][0]['base_position_m'][1],.2)
    def test_five_mm_tie_uses_smaller_base_y_but_triplets_refuse(self):
        s=snapshot();s['bodies']['red_2x3_0']['position']=[.1,.2,0.];s['bodies']['red_2x3_1']['position']=[.2,-.105,0.]
        p=visible_target(s,'red');self.assertTrue(p['ambiguous'])
        self.assertEqual(p['nearest_distance_object'],'red_2x3_0');self.assertEqual(p['object'],'red_2x3_1')
        with self.assertRaisesRegex(ValueError,'<=5 mm'):visible_target(s,'red',True)
        s['bodies']['red_2x3_0']['position']=[.1,0.,0.]
        for delta,ambiguous in ((.005-1e-9,True),(.005+1e-9,False)):
            s['bodies']['red_2x3_1']['position']=[.1+delta,0.,0.]
            self.assertEqual(visible_target(s,'red')['ambiguous'],ambiguous)
    def test_opt_in_changes_only_target_and_evidence_not_random_draws(self):
        for seed in range(30):
            original=parameters(seed,'red');before=copy.deepcopy(original)
            selected=apply_target_rule(original,snapshot())
            self.assertEqual(original,before)
            self.assertEqual({k:v for k,v in selected.items() if k not in ('object','target_selection')},
                {k:v for k,v in original.items() if k!='object'})
            self.assertEqual(selected['object'],'red_2x3_0');self.assertNotIn('target_selection',parameters(seed,'red'))
        with self.assertRaisesRegex(ValueError,'Unknown visible'):apply_target_rule(before,snapshot(),rule='random')
    def test_missing_or_nonfinite_candidate_or_base_refuses(self):
        for field in ('missing','body','base','quaternion'):
            s=snapshot()
            if field=='missing':s['bodies'].pop('red_2x3_1')
            elif field=='body':s['bodies']['red_2x3_0']['position'][0]=float('nan')
            elif field=='base':s['base_world_pose']['position'][0]=float('inf')
            else:s['base_world_pose']['orientation']=[0.,0.,0.,0.]
            with self.subTest(field=field),self.assertRaises((ValueError,KeyError)):visible_target(s,'red')


class AnalysisTests(unittest.TestCase):
    def fixture(self,root):
        name='rtx6000-normal-i3-b00';run=root/name/'normal';meta=run/'dataset/meta';meta.mkdir(parents=True)
        manifest=[];index=[]
        for i,color in enumerate(('red','green','blue')):
            p=parameters(530000+i,color);p.update(trial=i,object=f'{color}_2x3_{i%2}')
            raw=dict(params=p,accepted=True,episode_id='id-'+str(i),initial_physical=snapshot())
            attempt=run/f'attempt-{i:04d}';attempt.mkdir();(attempt/'receipt.json').write_text(json.dumps(raw))
            manifest.append(dict(episode_index=i,**raw));index.append(dict(batch=name,seed=p['seed'],color=color))
        (meta/'collection_episodes.json').write_text(json.dumps(manifest))
        (run/'summary.json').write_text(json.dumps(dict(complete=True,end_marker='COLLECTION_FINISHED',episodes=3)))
        (run/'collector-finalization.json').write_text(json.dumps(dict(completed=True,passed=True,cleanup_errors=[])))
        card=root/'rtx6000-normal-900-index';card.mkdir();(card/'data-card.json').write_text(json.dumps(dict(passed=True,accepted=3,episodes=index,batches=[dict(name=name,accepted=3)])))
        return run
    def test_all_episode_counts_per_color_batch_and_selected_seed_lists(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);summary,rows=analyze(root,expected=3)
            self.assertEqual(summary['counts']['visible_rule_matches'],2);self.assertEqual(summary['counts']['unambiguous_matches'],2)
            self.assertEqual([r['seed'] for r in rows if r['visible_rule_matches']],[530000,530002])
            self.assertEqual(summary['per_color']['green']['visible_rule_matches'],0)
            self.assertEqual(len(summary['per_batch']),1);self.assertEqual(summary['counts']['margin_mm']['count'],3)
    def test_edited_raw_or_duplicate_index_or_incomplete_batch_refuses(self):
        for change in ('raw','index','summary'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);run=self.fixture(root)
                path=run/'attempt-0000/receipt.json' if change=='raw' else root/'rtx6000-normal-900-index/data-card.json' if change=='index' else run/'summary.json'
                value=json.loads(path.read_text())
                if change=='raw':value['params']['object']='red_2x3_1'
                elif change=='index':value['episodes'][1]=value['episodes'][0]
                else:value['complete']=False
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):analyze(root,expected=3)


if __name__=='__main__':unittest.main()
