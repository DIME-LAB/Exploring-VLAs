"""Observe the collector's existing readiness wait without changing its budget."""
from contextlib import contextmanager
import datetime
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'collect'))


@contextmanager
def trace_wait_ready(collector, output, poll_s=.05):
    original = collector.wait_ready

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
            return original()
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
                          poll_s=poll_s, budget_changed=False, predicate_changed=False,
                          semantics='passive attribute reads; original wait_ready called exactly once; no ROS calls, app updates or physics steps')
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
