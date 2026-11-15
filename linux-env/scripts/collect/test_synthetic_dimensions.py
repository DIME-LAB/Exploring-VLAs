"""Use the seed-650009 native shape without the published bbox's rounding."""
import copy,json,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from recovery_start import native_bbox
from regrasp_runner import screen_evidence
from test_review_regressions import load_class


class NativeDimensionTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).with_name('fixtures')/'seed-650009-dimensions.json').read_text())

    def request(self,fixture,published):
        receipt=fixture['receipt'];before=receipt['recovery_report']['baseline'];old=receipt['recovery_report']['request']
        Collector,_=load_class('collect.py','Collector');c=object.__new__(Collector)
        c.writer=None;c.current_episode_id=receipt['episode_id'];c.args=NS(replay_receipt=None)
        c.generation=NS(epoch=before['epoch'],require=lambda *args:None)
        c.held_snapshot=lambda epoch:copy.deepcopy(before)
        c.node=NS(objects_lock=threading.RLock(),objects_data={receipt['params']['object']:old['pose']},
            _lookup_bbox=lambda name:published,_tcp_clearance_var=NS(get=lambda:1.),
            _grasp_cross_var=NS(get=lambda:False))
        class PlanningSceneBoundary(Exception):pass
        def boundary(**kwargs):raise PlanningSceneBoundary()
        c.sync_scene=boundary;result={}
        with self.assertRaises(PlanningSceneBoundary):c._prepare_regrasp_start(receipt['params'],result)
        return result['recovery_report']['request']

    def test_seed_650009_actual_request_entrance_replays_as_velocity_only(self):
        f=self.fixture();r=f['receipt'];target=r['params']['object']
        self.assertEqual(r['params']['seed'],650009)
        with self.assertRaisesRegex(RuntimeError,'Synthetic request dimensions differ'):
            screen_evidence(r,f['fault'],f['transport'])
        published=r['recovery_report']['request']['bbox']
        request=self.request(f,published)
        self.assertEqual(request['bbox'],native_bbox(r['recovery_report']['baseline'],target))
        self.assertEqual(request['published_bbox'],published)
        self.assertEqual(request['bbox_source'],'native_local_corners')
        self.assertGreater(abs(request['bbox']['sx']-published['sx']),1e-6)
        new=copy.deepcopy(r);new['recovery_report']['request']=request
        result=screen_evidence(new,f['fault'],f['transport'])
        self.assertEqual(result['reason'],'protected_velocity_screen')
        self.assertIn('cup_blue',result['bodies'])
        new['recovery_report']['request']['bbox']['sx']+=3e-6
        with self.assertRaisesRegex(RuntimeError,'synthetic.request.bbox'):
            screen_evidence(new,f['fault'],f['transport'])

    def test_missing_or_incorrect_published_bbox_never_controls_native_geometry(self):
        f=self.fixture();expected=native_bbox(f['receipt']['recovery_report']['baseline'],f['receipt']['params']['object'])
        for published in (None,dict(sx=1.,sy=1.,sz=1.)):
            with self.subTest(published=published):
                request=self.request(f,published)
                self.assertEqual(request['bbox'],expected);self.assertEqual(request['published_bbox'],published)

    def test_invalid_native_corners_refuse_without_published_fallback(self):
        f=self.fixture();target=f['receipt']['params']['object'];snapshot=f['receipt']['recovery_report']['baseline']
        for corners in ([],[[0.,0.,0.]]*8,[[float('nan'),0.,0.]]*8):
            bad=copy.deepcopy(snapshot);bad['local_corners'][target]=corners
            with self.subTest(corners=corners),self.assertRaises(ValueError):native_bbox(bad,target)


if __name__=='__main__':unittest.main()
