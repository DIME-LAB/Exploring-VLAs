"""Physical quality outcomes and worker retirement, independent of ROS transport."""
import math
import threading
import time
from protocol import CancelledGeneration
from recovery import receipt_error

POLICY = dict(version=1, consecutive_same_reason_limit=3,
              convergence_tolerance_rad=.008, home_tolerance_rad=.01,
              home_spread_rad=.002, home_settle_sim_seconds=5.)
QUALITY_KINDS = frozenset(('arm_nonconvergence', 'gripper_nonconvergence', 'home_unstable',
                          'planner_reject', 'containment', 'insufficient_lift',
                          'stationary', 'short_duration','regrasp_miss_not_confirmed'))
PLANNER_REJECTIONS = frozenset(('grasp_unreachable_workspace','grasp_unreachable_ik',
                                'gate_c_all_collide','tiered_planner_exhausted','ompl_mode_b'))


class QualityReject(RuntimeError):
    def __init__(self, kind, evidence):
        if kind not in QUALITY_KINDS:
            raise ValueError('Unknown physical quality rejection')
        self.kind, self.evidence = kind, evidence
        super().__init__(kind+': '+repr(evidence))


class ConvergenceFailure(TimeoutError):
    """Only Servo.tick raises this, after joined feedback and a sim deadline."""
    def __init__(self, kind, *, epoch, stamp_ns, phase, duration, elapsed, margin, state, reference):
        if kind not in ('arm_nonconvergence','gripper_nonconvergence'):
            raise ValueError('Unknown motion convergence failure')
        if epoch is None or elapsed <= duration+margin or not all(math.isfinite(q) for q in [*state,*reference]):
            raise ValueError('Convergence failure lacks valid physical deadline evidence')
        self.kind = kind
        self.evidence = dict(kind=kind, epoch=epoch, stamp_ns=stamp_ns, phase=phase,
            planned_sim_seconds=duration, elapsed_sim_seconds=elapsed, margin_sim_seconds=margin,
            state=list(state), reference=list(reference))
        super().__init__(kind+': '+repr(self.evidence))


class WorkerRegistry:
    """Capture worker origins at creation; retire their threads before another attempt."""
    def __init__(self, generation, spawn, token):
        self.generation, self.spawn, self.token = generation, spawn, token
        self.lock = threading.RLock()
        self.workers = []

    def thread(self, target, **kwargs):
        epoch = self.token()
        with self.generation.own('motion worker registration'):
            self.generation.require(epoch)
            def checked():
                self.generation.require(epoch)
                target()
            worker = self.spawn(target=checked, **kwargs)
            with self.lock:
                self.workers.append((epoch, worker))
            return worker

    def retire(self, timeout=15.):
        end = time.monotonic()+timeout
        while True:
            with self.lock:
                entries = list(self.workers)
            pending = []
            for epoch, worker in entries:
                if worker is threading.current_thread():
                    raise RuntimeError('Motion worker cannot retire itself')
                if worker.ident is None or worker.is_alive():
                    pending.append((epoch,worker))
            if not pending:
                with self.lock:
                    if entries != self.workers:
                        continue
                    self.workers.clear()
                return dict(passed=True, workers=len(entries), epochs=sorted({e for e,_ in entries}),
                            native_barrier_epoch=self.generation.epoch)
            if time.monotonic() >= end:
                raise RuntimeError('Motion workers did not retire: '+repr([(e,w.name,w.ident) for e,w in pending]))
            for _, worker in pending:
                if worker.ident is not None:
                    worker.join(timeout=min(.05,max(0.,end-time.monotonic())))
            threading.Event().wait(.002)


def disposition(receipt, fatal=None):
    """Only positive physical evidence admits another attempt; other faults stop."""
    if fatal:
        return dict(kind='terminal',reason='pipeline_fault',error=str(fatal))
    native=receipt.get('end_cancel',{})
    if not native.get('held') or native.get('failure'):
        return dict(kind='terminal',reason='native_cancel')
    if not receipt.get('retirement',{}).get('passed'):
        return dict(kind='terminal',reason='worker_retirement')
    timing = receipt.get('timing')
    if timing and not timing.get('contract_passed', timing.get('passed',False)):
        return dict(kind='terminal',reason='audit_contract',errors=timing.get('contract_errors',timing.get('errors')))
    quality = receipt.get('quality_reject')
    if quality and quality.get('kind') in QUALITY_KINDS:
        return dict(kind='quality_reject',reason=quality['kind'])
    if receipt.get('original_capture_error_preserved'):
        for record in reversed(receipt.get('capture_failures',[])):
            physical = record.get('physical_convergence')
            if (physical and physical['kind'] in ('arm_nonconvergence','gripper_nonconvergence') and
                    record['cancellation'].get('completed') and
                    record['cancellation'].get('result',{}).get('held') and
                    not record['cancellation'].get('result',{}).get('failure') and
                    record['epoch']==physical['epoch'] and
                    record.get('validated_sim_tick') is True):
                return dict(kind='quality_reject',reason=physical['kind'])
    if receipt.get('accepted') and receipt_error(receipt):
        return dict(kind='terminal',reason='missing_regrasp_evidence',error=receipt_error(receipt))
    if receipt.get('accepted') and timing and timing.get('passed'):
        return dict(kind='accepted',reason='success')
    if receipt.get('script_completed') and receipt.get('score',{}).get('passed') and timing:
        errors = set(timing.get('quality_errors',[]))
        if errors and errors <= {'stationary','short_duration'}:
            return dict(kind='quality_reject',reason='stationary' if 'stationary' in errors else 'short_duration')
    return dict(kind='terminal',reason='unclassified_failure',error=receipt.get('error'))


class FailureStreak:
    def __init__(self, receipts=()):
        self.reason, self.count, self.stopped = None, 0, False
        for receipt in receipts:
            if 'disposition' not in receipt:
                raise ValueError('Receipt predates the quality policy; use a fresh output directory')
            self.observe(receipt['disposition'])

    def observe(self, decision):
        if self.stopped:
            raise ValueError('Receipt ledger continued after a mandatory stop')
        if decision['kind']=='accepted':
            self.reason,self.count=None,0
        elif decision['kind']=='quality_reject':
            self.count=self.count+1 if self.reason==decision['reason'] else 1
            self.reason=decision['reason']
        else:
            self.stopped=True
            return dict(stop=True,reason='terminal',count=self.count)
        self.stopped=self.count>=POLICY['consecutive_same_reason_limit']
        return dict(stop=self.stopped,reason=self.reason,count=self.count)
