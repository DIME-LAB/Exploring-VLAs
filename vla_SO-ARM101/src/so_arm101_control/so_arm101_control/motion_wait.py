"""Operation-owned simulation deadlines with a separate bounded wall watchdog."""
import math
import threading
import time


class MotionWaitTimeout(TimeoutError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(str(diagnostic))


class MotionBudget:
    def __init__(self, epoch, parent=None):
        self._epoch = epoch
        self._planned = 0.
        self._children = []
        self._lock = threading.Lock()
        if parent is not None:
            if parent.epoch != epoch:
                raise ValueError('Motion timing parent has a different origin')
            with parent._lock:
                parent._children.append(self)

    @property
    def epoch(self):
        return self._epoch

    def plan(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Motion duration must be finite and positive')
        with self._lock:
            self._planned = max(self._planned, float(seconds))

    def total(self):
        with self._lock:
            planned, children = self._planned, list(self._children)
        totals = [child.total() for child in children]
        return planned + sum(t[0] for t in totals), 1 + sum(t[1] for t in totals)


class MotionWaiter:
    def __init__(self, clock, validate, emit, abort=lambda:False,
                 sim_margin=5., rtf_floor=.1, wall_margin=30., wall=time.monotonic):
        if not all(math.isfinite(v) and v>0 for v in (sim_margin,rtf_floor,wall_margin)):
            raise ValueError('Motion wait policy requires positive bounds')
        self.clock, self.validate, self.emit, self.abort = clock, validate, emit, abort
        self.sim_margin, self.rtf_floor, self.wall_margin, self.wall = sim_margin, rtf_floor, wall_margin, wall

    def wait_until(self, ready, park, budget, epoch, wall_min=0., label='motion', enabled=lambda:True):
        start_sim, start_wall = self.clock(), self.wall()
        while True:
            self.validate(epoch)
            aborted = self.abort()
            if not aborted and enabled() and ready():
                self.validate(epoch)
                return True
            planned, operations = budget()
            sim_limit = planned + self.sim_margin*max(1,operations)
            wall_limit = max(wall_min, sim_limit/self.rtf_floor + self.wall_margin)
            elapsed_sim, elapsed_wall = self.clock()-start_sim, self.wall()-start_wall
            kind = ('aborted' if aborted else
                    'clock_reversed' if elapsed_sim < -1e-6 else
                    'sim_deadline' if elapsed_sim > sim_limit else
                    'wall_watchdog' if elapsed_wall > wall_limit else None)
            if kind:
                self.validate(epoch)
                record = dict(event='motion_wait_timeout', kind=kind, label=label, epoch=epoch,
                    planned_sim_s=planned, operations=operations, sim_margin_s=self.sim_margin,
                    elapsed_sim_s=elapsed_sim, elapsed_wall_s=elapsed_wall,
                    sim_limit_s=sim_limit, wall_watchdog_s=wall_limit, rtf_floor=self.rtf_floor)
                try:
                    self.emit(record)
                finally:
                    raise MotionWaitTimeout(record)
            park(.05)
            self.validate(epoch)

    def wait(self, event, wall_min=0., enabled=lambda:True):
        if event.motion_budget.epoch != event.motion_epoch:
            raise ValueError('Motion timing budget and event origins differ')
        return self.wait_until(event.is_set, lambda dt:event.wait(timeout=dt) if enabled() else time.sleep(dt),
            event.motion_budget.total, event.motion_epoch, wall_min,
            getattr(event,'motion_tag','motion'), enabled)

    def wait_thread(self, thread, budget, epoch, wall_min):
        return self.wait_until(lambda:not thread.is_alive(), lambda dt:thread.join(timeout=dt),
                               budget, epoch, wall_min, 'episode')
