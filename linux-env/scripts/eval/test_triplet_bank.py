"""CPU admission tests against the actual sealed held registry and producer."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import triplet_bank as bank

ROOT = Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval/triplet360/held20-v2')
BANK_SHA = '0ac0d8928a960ce0aa22670f2d75c33bbc9a2b6717ba3f0fc076bec652abab0d'
COLLECTOR = Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval-native-triplet360')


@unittest.skipUnless(ROOT.exists(), 'actual sealed dual triplet bank required')
class TripletAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(COLLECTOR/'linux-env/scripts/collect'))
        cls.rows, cls.suite = bank.load(ROOT, BANK_SHA)

    def test_actual_registry_membership_and_replacements(self):
        self.assertEqual(len(self.rows),60)
        actual = [r['layout_seed'] for r in self.rows[::3]]
        self.assertEqual(len(set(actual)),20)
        self.assertEqual([r['requested_layout_seed'] for r in self.rows[::3]],list(range(730000,730020)))
        self.assertEqual(actual[6],730020)
        self.assertEqual(actual[8],730021)
        self.assertEqual(actual[12],730023)
        self.assertEqual([r['color'] for r in self.rows],list(bank.COLORS)*20)

    def test_shared_probe_images_and_noise(self):
        import numpy as np
        from flow_noise import episode_identity
        homes=bank.document(ROOT/'held20-inputs.json')
        self.assertEqual(len(homes),20)
        for i,home in enumerate(homes):
            with np.load(ROOT/home['cameras']) as pixels:
                self.assertEqual(set(pixels.files),{'top','wrist'})
                for key in pixels.files:
                    self.assertEqual(pixels[key].shape,(480,640,3))
                    self.assertEqual(pixels[key].dtype,np.uint8)
            identities=[episode_identity(r['layout_seed'],0) for r in self.rows[3*i:3*i+3]]
            self.assertEqual(identities,[episode_identity(home['seed'],0)]*3)
            self.assertEqual(set(home['expert_by_prompt']),set(bank.COLORS))

    def test_fixed_complete_rgb_chunks_and_legacy_unchanged(self):
        from native_bank import select_policy_rows
        for offset in (0,15,30,45):
            chunk=select_policy_rows(self.rows,self.suite,offset,15)
            self.assertEqual(len(chunk),15)
            self.assertEqual(len(set(r['layout_seed'] for r in chunk)),5)
        for offset,count in ((1,15),(0,14),(45,16),(60,15)):
            with self.assertRaises(ValueError):select_policy_rows(self.rows,self.suite,offset,count)
        old=[dict(seed=i) for i in range(5)]
        self.assertEqual(select_policy_rows(old,'seen-training-home'),old)

    def test_wrong_bank_sha_refused(self):
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            bank.load(ROOT,'0'*64)

    def test_private_runtime_matches_candidate_qualification_sources(self):
        from native_environment import qualification_sources
        dev=Path('/mnt/extreme4tb/aaugus11/exploring-vlas/eval-triplet360-dev/linux-env/scripts/eval')
        self.assertEqual(qualification_sources(dev),qualification_sources(COLLECTOR/'linux-env/scripts/eval'))

    def test_producer_reset_exact_sequence_no_sampler(self):
        """Real producer preparation; fake native RPC returns recorded proof shapes."""
        frozen_root=Path(json.loads((COLLECTOR/'linux-env/scripts/collect/hosts/dual-a4500.json').read_text())['root'])
        if not ROOT.is_relative_to(frozen_root):
            self.skipTest(f'frozen native collector {COLLECTOR} confines writes to {frozen_root}, but the held bank is {ROOT} '
                          '(2026-11-25 storage switch; the old root is read-only). Needs a re-cut collector on the new root.')
        from triplet_reset import install_recorded_session,require_bootstrap
        from target_triplets import restore_code
        row=self.rows[0];source=bank.document(row['source_receipt']);anchor=bank.document(row['anchor_path'])
        events=[]
        priming=source['counterfactual']['priming_restore']
        final=dict(source['initial_restore_check'],snapshot=source['initial_physical'])
        restores=iter((priming,final))
        def restore(code):
            self.assertEqual(code,restore_code(anchor['anchor'],True))
            events.append('restore');return copy.deepcopy(next(restores))
        def advance(count):
            events.append(('advance',count))
            self.assertIn(count,(20,1))
            return [copy.deepcopy(source['episode_start_physical'])] * count
        node=SimpleNamespace(objects_data={},_drop_data={},_grasp_approach_height_var=SimpleNamespace(set=lambda v:None))
        collector=SimpleNamespace(writer=None,timeline=lambda value:events.append(('timeline',value)),
            scene_reset=SimpleNamespace(clear=lambda:{}),gpu_footprint=lambda:{'passed':True},camera_graph=lambda:{},
            restore_physical=restore,advance_ticks=advance,servo=SimpleNamespace(configure=lambda params:events.append('configure')),
            generation=SimpleNamespace(own=lambda why:nullcontext(),require=lambda epoch:None),
            sync_scene=lambda **kw:None,update_pose_cache=lambda snapshot:None,
            mutate_kit=lambda *args:{name:True for name in (anchor['target_selections'][c]['object'] for c in bank.COLORS)},
            node=node,scene_pose_check={},scene_clear_check={},sensors=SimpleNamespace(duplicate_messages={}))
        with tempfile.TemporaryDirectory(prefix='triplet-reset-',dir=ROOT.parent) as temporary:
            params=install_recorded_session(collector,row,temporary)
            receipt=dict(episode_id=source['episode_id'])
            collector.triplet_session.prepare(collector,params,{},receipt,set(),temporary,Path(temporary)/'attempt-0000')
            self.assertEqual(events,[('timeline',False),'restore','configure',('advance',20),'restore',('advance',1),'configure'])
            self.assertEqual(collector.triplet_session.sampled,0)
            self.assertEqual(params,source['params'])
            require_bootstrap(collector.episode_start_physical,source['episode_start_physical'])
            self.assertEqual(receipt['counterfactual']['sampler_invocations'],0)
            bad=copy.deepcopy(collector.episode_start_physical);bad['state'][0]+=.001
            with self.assertRaises(ValueError):require_bootstrap(bad,source['episode_start_physical'])


if __name__=='__main__':
    unittest.main()
