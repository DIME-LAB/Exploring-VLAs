"""Observe collector readiness with a separately approved opt-in deadline."""
from contextlib import contextmanager
import datetime
from pathlib import Path
import sys
import threading
import time
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'collect'))


def budgeted_wait_ready(original, budget_s=60):
    """Keep the frozen collector method's bytecode; opt in to its longer deadline."""
    if budget_s not in (60, 180):
        raise ValueError('Readiness budget must be 60 or explicitly approved 180 seconds')
    if budget_s == 60:
        return original
    function = getattr(original, '__func__', None)
    if function is None or function.__code__.co_name != 'wait_ready':
        raise ValueError('Budget override requires the bound collector wait_ready method')
    constants = function.__code__.co_consts
    indices = [i for i, value in enumerate(constants) if type(value) is int and value == 60]
    if len(indices) != 1:
        raise ValueError('Frozen wait_ready must have exactly one 60-second deadline constant')
    changed = list(constants)
    changed[indices[0]] = int(budget_s)
    code = function.__code__.replace(co_consts=tuple(changed))
    replacement = types.FunctionType(code, function.__globals__, function.__name__,
                                     function.__defaults__, function.__closure__)
    return types.MethodType(replacement, original.__self__)


@contextmanager
def trace_wait_ready(collector, output, poll_s=.05, readiness_budget_s=60):
    original = collector.wait_ready
    waiter = budgeted_wait_ready(original, readiness_budget_s)

    def observed_wait():
        started = time.monotonic()
        stop = threading.Event()
        events, failures, first_ready = [], [], {}
        primary = None

        def observe():
            try:
                ready = dict(objects_data=bool(collector.node.objects_data),
                             three_cups=len(collector.node._drop_data) == 3,
                             servo_last_sim_ns=bool(collector.servo.last_sim_ns),
                             sensors_reference=bool(collector.sensors.reference))
                elapsed = time.monotonic()-started
                for key, value in ready.items():
                    if value and key not in first_ready:
                        first_ready[key] = elapsed
                if not events or ready != events[-1]['ready']:
                    events.append(dict(elapsed_s=elapsed, ready=ready,
                                       missing=[key for key, value in ready.items() if not value]))
            except Exception as error:
                failures.append(dict(elapsed_s=time.monotonic()-started, error=repr(error)))

        def monitor():
            while not stop.wait(poll_s):
                observe()

        observe()
        thread = threading.Thread(target=monitor, name='eval-readiness-observer', daemon=True)
        thread.start()
        try:
            return waiter()
        except BaseException as error:
            primary = error
            raise
        finally:
            stop.set()
            thread.join()
            observe()
            result = dict(status='PASS' if primary is None else 'FAILED',
                          observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          elapsed_s=time.monotonic()-started, first_observed_ready_s=first_ready,
                          missing_at_end=events[-1]['missing'] if events else None,
                          transitions=events, observer_errors=failures,
                          error=repr(primary) if primary is not None else None,
                          poll_s=poll_s, readiness_budget_s=readiness_budget_s,
                          budget_changed=readiness_budget_s != 60, predicate_changed=False,
                          semantics='passive attribute reads; same collector wait body called once; explicit deadline only; no observer ROS calls, app updates or physics steps')
            try:
                from storage import atomic_json
                atomic_json(output, result)
            except Exception:
                if primary is None:
                    raise

    collector.wait_ready = observed_wait
    try:
        yield
    finally:
        collector.wait_ready = original
