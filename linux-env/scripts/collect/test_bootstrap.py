"""Constructor-only status ownership; epoch=None never becomes a motion epoch."""
import threading,unittest
from test_review_regressions import load_class
from protocol import StepGeneration,CancelledGeneration


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_only_the_constructor_thread_and_is_revoked(self):
        Collector,_=load_class('collect.py','Collector')
        c=object.__new__(Collector);c.generation=StepGeneration();c._constructing_thread=threading.get_ident()
        with c.motion_effect(None):pass
        errors=[]
        def stale_worker():
            try:
                with c.motion_effect(None):errors.append('accepted')
            except CancelledGeneration:errors.append('refused')
        t=threading.Thread(target=stale_worker);t.start();t.join(2)
        self.assertEqual(errors,['refused'])
        c._constructing_thread=None
        with self.assertRaises(CancelledGeneration):
            with c.motion_effect(None):pass
        c.generation.install(dict(epoch=1,invalid_through_ns=0))
        with c.motion_effect(1):pass
        with self.assertRaises(CancelledGeneration):
            with c.motion_effect(None):pass

    def test_poisoned_or_installed_generation_cannot_use_bootstrap(self):
        Collector,_=load_class('collect.py','Collector')
        c=object.__new__(Collector);c.generation=StepGeneration();c._constructing_thread=threading.get_ident()
        c.generation.install(dict(epoch=1,invalid_through_ns=0))
        with self.assertRaises(CancelledGeneration):
            with c.motion_effect(None):pass
        c.generation.poison(RuntimeError('stopped'))
        with self.assertRaises(CancelledGeneration):
            with c.motion_effect(None):pass
