"""Ordered reserve allocation and the approved production refusal guards."""
import copy,hashlib,random,tempfile,unittest
from pathlib import Path
from contract import parameters
from target_triplets import spec
from triplet_production_plan import initial_wave,reduce_waves,guards,BASE,RESERVE,validate_ranges
import test_target_triplets


def outcome(j,reason='accepted'):
    return dict(split=j['split'],requested_layout_seed=j['requested_layout_seed'],layout_seed=j['layout_seed'],
        status='selected' if reason=='accepted' else 'refused',reason=reason,
        proof_path='/verified/'+str(j['layout_seed'])+'.json',proof_sha256=hashlib.sha256(str(j['layout_seed']).encode()).hexdigest())


class TripletProductionPlanTests(unittest.TestCase):
    def test_base_plan_and_all_reserved_ranges_are_disjoint(self):
        self.assertTrue(validate_ranges());jobs=initial_wave();self.assertEqual(len(jobs),140)
        self.assertEqual({s:sum(j['split']==s for j in jobs) for s in RESERVE},dict(train=120,heldout=20))
        self.assertEqual([sum(j['instance']==i for j in jobs) for i in (3,4,5,6)],[35]*4)
        for seed,split in ((720120,'train'),(720199,'train'),(730020,'heldout'),(730049,'heldout')):
            self.assertEqual(spec(seed,split)['layout_seed'],seed)
            with self.assertRaises(ValueError):parameters(seed,'red',regrasp=True,split=split)
        for seed,split in ((720200,'train'),(730050,'heldout'),(730020,'train'),(720120,'heldout')):
            with self.assertRaises(ValueError):spec(seed,split)
    def test_sorted_wave_replacements_ignore_completion_order_and_preserve_chains(self):
        jobs=initial_wave();failed={720002,720005,730005}
        results=[outcome(j,'visible_target_margin' if j['layout_seed'] in failed else 'accepted') for j in jobs]
        history=[dict(jobs=jobs,outcomes=results)];state=reduce_waves(history)
        self.assertEqual([(j['requested_layout_seed'],j['layout_seed']) for j in state['next_wave']],[(720002,720120),(720005,720121),(730005,730020)])
        random.Random(7).shuffle(results);self.assertEqual(state,reduce_waves(history))
        wave=state['next_wave'];history.append(dict(jobs=wave,outcomes=[outcome(j,'quality:containment' if j['layout_seed']==720120 else 'accepted') for j in wave]))
        state=reduce_waves(history);self.assertEqual(state['next_wave'][0]['replacement_chain'],[720002,720120,720122])
        history.append(dict(jobs=state['next_wave'],outcomes=[outcome(state['next_wave'][0])]))
        state=reduce_waves(history);self.assertTrue(state['complete']);self.assertEqual(len(state['selected']),140)
        self.assertEqual(len(state['failed_groups']),4)
        actual=[j['layout_seed'] for j in state['selected']];self.assertEqual(len(set(actual)),140)
    def test_empty_or_edited_checkpoint_cannot_adopt_selected_members(self):
        state=reduce_waves([]);self.assertFalse(state['complete']);self.assertEqual(state['selected'],[])
        jobs=initial_wave();results=[outcome(j) for j in jobs]
        for edit in ('missing','duplicate','orphan','slot','flag','jobs','proof'):
            wave=copy.deepcopy(dict(jobs=jobs,outcomes=results))
            if edit=='missing':wave['outcomes'].pop()
            elif edit=='duplicate':wave['outcomes'].append(wave['outcomes'][0])
            elif edit=='orphan':wave['outcomes'][0]['layout_seed']=720120
            elif edit=='slot':wave['outcomes'][0]['requested_layout_seed']=720001
            elif edit=='flag':wave['outcomes'][0]['reason']='exact_identity'
            elif edit=='jobs':wave['jobs'][0]['layout_seed']=720120
            else:wave['outcomes'][0]['proof_sha256']='unverified'
            with self.subTest(edit=edit),self.assertRaises(ValueError):reduce_waves([wave])
    def test_operator_identity_and_half_reserve_stop_guards_have_no_minimum_window(self):
        with self.assertRaisesRegex(ValueError,'exceed 10%'):guards([dict(reason='exact_identity')],[])
        guards([dict(reason='accepted')]*9+[dict(reason='exact_identity')],[])
        with self.assertRaisesRegex(ValueError,'exceed 10%'):guards([dict(reason='accepted')]*8+[dict(reason='exact_identity')]*2,[])
        for split,limit in (('train',40),('heldout',15)):
            guards([], [dict(requested_layout_seed=BASE[split][0])]*limit)
            with self.assertRaisesRegex(ValueError,'half'):guards([], [dict(requested_layout_seed=BASE[split][0])]*(limit+1))
    def test_canonical_native_snapshot_survives_visible_rule_refusal(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            h=test_target_triplets.TripletTests().fixture(d)
            h.data['bodies']['red_2x3_1']['position']=list(h.data['bodies']['red_2x3_0']['position'])
            with self.assertRaisesRegex(ValueError,'<=5 mm'):h.prepare(0)
            doc=json.loads((h.root/'triplet-boundaries/attempt-0000.json').read_text())
            self.assertIn('snapshot',doc['phases']['canonical_restore']['restore'])
            self.assertFalse((h.root/'attempt-0000').exists())


if __name__=='__main__':unittest.main()
