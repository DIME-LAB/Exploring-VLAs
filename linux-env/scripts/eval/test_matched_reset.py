import hashlib,json,tempfile,unittest
import collections,threading,types,sys,math,copy
from pathlib import Path
from matched_reset_client import require_qualification,install_fenced_boundary
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collect'))
from protocol import StepGeneration
from matched_reset import orientation_errors
from native_bank import SEEN_BANK_SHA
class Tests(unittest.TestCase):
    def test_cup_geodesic_limit_does_not_relax_lego_limit(self):
        expected=dict(bodies={n:dict(orientation=[1.,0.,0.,0.]) for n in ('cup_red','red_2x3_0')});actual=copy.deepcopy(expected)
        def rotate(name,degrees):actual['bodies'][name]['orientation']=[math.cos(math.radians(degrees)/2),0.,0.,math.sin(math.radians(degrees)/2)]
        rotate('cup_red',.08);self.assertTrue(orientation_errors(actual,expected)['passed'])
        rotate('cup_red',.11);self.assertFalse(orientation_errors(actual,expected)['passed'])
        rotate('cup_red',.08);rotate('red_2x3_0',.02);self.assertFalse(orientation_errors(actual,expected)['passed'])
    def test_pending_joins_are_retired_before_metadata_is_pruned(self):
        generation=StepGeneration();generation.install(dict(epoch=1,invalid_through_ns=100))
        sensors=types.SimpleNamespace(condition=threading.Condition(),pending=collections.OrderedDict([(110,dict(wrist='old',top='old',tick='old',joint='old'))]),acks={(1,110):'old'})
        def prune():
            self.assertFalse(sensors.pending,'A complete stale join would reference pruned camera metadata')
            self.assertFalse(sensors.acks)
        sensors.prune_durable_camera_history=prune
        transport=types.SimpleNamespace(generation=generation,sensors=sensors,session=types.SimpleNamespace(current='old',entry='old'))
        def mutation():
            self.assertIsNone(generation.epoch,'Camera identity must be invalid before clearing buffered joins')
            self.assertFalse(sensors.pending)
            return dict(epoch=2,invalid_through_ns=300)
        result,retired=install_fenced_boundary(transport,mutation)
        self.assertEqual(generation.identity,(2,300));self.assertEqual(retired,dict(pending_groups=1,acks=1));self.assertIsNone(transport.session.current)
    def test_heldout_requires_all_ordered_seeds_and_an_unchanged_bootstrap_control(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);sources=base/'sources';sources.mkdir();hashes={}
            for name in ('matched_reset.py','matched_reset_client.py','native_bank.py','qualify_matched_reset.py'):
                (sources/name).write_text(name);hashes[name]=hashlib.sha256(name.encode()).hexdigest()
            host=dict(name='dual-a4500',root='/backup');collector='/backup/collector'
            def write(root,suite,bank,control=None):
                (root/'qualification').mkdir(parents=True)
                seeds=list(range(530000,530005)) if suite=='seen-training-home' else list(range(900000,900030))
                ledger=dict(status='matched-reset-qualified',cleanup_exit_code=0,host_profile=host,source_sha256=hashes)
                summary=dict(status='PASS',suite=suite,seeds=seeds,reference_run_sha256='reference',bootstrap_control=control,episodes=len(seeds),commands=10*len(seeds),bank_sha256=bank,reset_collector_root=collector)
                (root/'window-ledger.json').write_text(json.dumps(ledger));(root/'qualification/summary.json').write_text(json.dumps(summary))
                return summary
            seen=base/'seen';write(seen,'seen-training-home',SEEN_BANK_SHA)
            control=require_qualification(seen,host,collector,SEEN_BANK_SHA,sources,reference_sha256='reference')
            held=base/'held';original=write(held,'heldout-bank-v2','held-bank',control)
            require_qualification(held,host,collector,'held-bank',sources,suite='heldout-bank-v2',reference_sha256='reference')
            for update in (dict(episodes=29),dict(commands=290),dict(seeds=list(reversed(original['seeds']))),dict(suite='seen-training-home'),dict(reference_run_sha256='wrong')):
                altered=dict(original,**update);(held/'qualification/summary.json').write_text(json.dumps(altered))
                with self.assertRaises(AssertionError):require_qualification(held,host,collector,'held-bank',sources,suite='heldout-bank-v2',reference_sha256='reference')
            altered=copy.deepcopy(original);altered['bootstrap_control']['bank_sha256']='another-seen-bank'
            (held/'qualification/summary.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(AssertionError,'fixed training-home bank'):require_qualification(held,host,collector,'held-bank',sources,suite='heldout-bank-v2')
            (held/'qualification/summary.json').write_text(json.dumps(original))
            (seen/'window-ledger.json').write_text((seen/'window-ledger.json').read_text()+' ')
            with self.assertRaisesRegex(AssertionError,'Bootstrap control changed'):require_qualification(held,host,collector,'held-bank',sources,suite='heldout-bank-v2')
    def test_changed_reset_source_cannot_use_a_previous_live_qualification(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'qualification').mkdir();sources=root/'source';sources.mkdir();hashes={}
            for name in ('matched_reset.py','matched_reset_client.py','native_bank.py','qualify_matched_reset.py'):
                (sources/name).write_text(name);hashes[name]=hashlib.sha256(name.encode()).hexdigest()
            host=dict(name='dual-a4500',root='/backup');collector='/backup/collector'
            ledger=dict(status='matched-reset-qualified',cleanup_exit_code=0,host_profile=host,source_sha256=hashes)
            summary=dict(status='PASS',suite='seen-training-home',seeds=list(range(530000,530005)),reference_run_sha256='reference',bootstrap_control=None,episodes=5,commands=50,bank_sha256='bank',reset_collector_root=collector)
            (root/'window-ledger.json').write_text(json.dumps(ledger));(root/'qualification/summary.json').write_text(json.dumps(summary))
            require_qualification(root,host,collector,'bank',sources)
            (sources/'matched_reset.py').write_text('changed')
            with self.assertRaisesRegex(AssertionError,'differs from live'):require_qualification(root,host,collector,'bank',sources)
if __name__=='__main__':unittest.main()
