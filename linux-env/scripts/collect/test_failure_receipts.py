"""Original collector failures retain epoch and cancellation provenance."""
import unittest
from types import SimpleNamespace as NS
import test_review_regressions as probes
from protocol import CancelledGeneration
from episode_policy import ConvergenceFailure


class FailureReceiptTests(unittest.TestCase):
    def test_capture_timeout_keeps_original_cause_and_native_cancel_result(self):
        c,r,entry=probes.ReviewRegressions().collector()
        def timeout(stamp,state,*args):
            c.servo.last_sim_ns=stamp
            raise ConvergenceFailure('arm_nonconvergence',epoch=1,stamp_ns=stamp,phase='move',
                duration=2.,elapsed=7.7,margin=5.,state=state,reference=c.servo.reference)
        c.servo.tick=timeout
        c.capture()
        self.assertEqual(len(c.capture_failures),1)
        record=c.capture_failures[0]
        self.assertEqual(record['epoch'],1);self.assertEqual(record['phase'],'move')
        self.assertEqual(record['exception_type'],'ConvergenceFailure')
        self.assertEqual(record['planned_sim_seconds'],2.)
        self.assertEqual(record['convergence_margin_sim_seconds'],5.)
        self.assertEqual(record['operation_start_sim_ns'],10**9)
        self.assertEqual(record['cancellation']['invalidated_epoch'],1)
        self.assertEqual(record['cancellation']['cause'],record['error'])
        self.assertEqual(record['measured_state'],[0.]*6)
        self.assertEqual(record['expert_reference'],[.1]*5+[0.])
        self.assertTrue(record['cancellation']['completed'])
        self.assertEqual(record['cancellation']['result']['epoch'],2)
        error=c.attempt_error(CancelledGeneration('cleanup retired epoch'),c.capture_failures,{1},2)
        self.assertEqual(error['error'],record['error'])
        self.assertIn('CancelledGeneration',error['cleanup_error'])
        self.assertTrue(error['original_capture_error_preserved'])
        self.assertNotIn('accepted',error)

    def test_unrelated_cancellation_and_failed_cancel_are_not_hidden(self):
        c,_,_=probes.ReviewRegressions().collector()
        record=dict(epoch=1,error='original timeout',cancellation=dict(completed=True,result=dict(epoch=2)))
        for epochs,current in [({9},2),({1},3),({1},None)]:
            exc=CancelledGeneration('unrelated epoch')
            self.assertEqual(c.attempt_error(exc,[record],epochs,current),dict(error=repr(exc)))
        record['cancellation']=dict(requested=True,completed=False,error='transport failed')
        exc=CancelledGeneration('cancel did not complete')
        self.assertEqual(c.attempt_error(exc,[record],{1},2),dict(error=repr(exc)))

    def test_scene_reset_transport_and_clock_faults_keep_their_own_error(self):
        c,_,_=probes.ReviewRegressions().collector()
        record=dict(epoch=1,error='old convergence',cancellation=dict(completed=True,result=dict(epoch=2)))
        for exc in [RuntimeError('clock jumped'),RuntimeError('reset mismatch'),ConnectionError('transport'),RuntimeError('scene incomplete')]:
            self.assertEqual(c.attempt_error(exc,[record],{1},2),dict(error=repr(exc)))
if __name__=='__main__':unittest.main()
