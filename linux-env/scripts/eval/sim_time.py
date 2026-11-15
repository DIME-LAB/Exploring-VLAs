"""Opt-in evaluation timing; production LeRobot transport/model paths stay intact.

The action clock is /clock when use_sim_time is true, monotonic wall time
otherwise. No global time module or installed fork is modified.
"""
from __future__ import annotations

import math
import threading
import time
from queue import Empty, Queue
from types import MethodType


class TickSchedule:
    """Absolute deadlines with no catch-up bursts after a skipped clock slot."""
    def __init__(self, start, fps):
        if fps <= 0:
            raise ValueError('Positive FPS required')
        self.start, self.period, self.slot = start, 1 / fps, 0

    @property
    def deadline(self):
        return self.start + self.slot * self.period

    def take(self, now):
        if now + 1e-8 < self.deadline:
            raise ValueError('Tick precedes deadline')
        missed = max(0, math.floor((now - self.deadline + 1e-8) / self.period))
        expected = self.deadline + missed * self.period
        self.slot += missed + 1
        return expected, missed


def delay_steps(previous_prediction_s, capture_s, now, fps, available):
    """Previous measured prediction plus current capture age, all in one clock."""
    # Separate /clock subscribers can be one frame apart while DDS catches up.
    if capture_s > now + 2/fps + 1e-6:
        raise ValueError('Observation is ahead of the selected clock')
    return min(available, max(0, math.ceil((previous_prediction_s + max(0, now-capture_s))*fps-1e-8)))


class PacingClock:
    def __init__(self, use_sim_time, node=None, stale_wall_s=2.0):
        self.use_sim_time = use_sim_time
        self.condition = threading.Condition()
        self.value = None
        self.error = None
        self.received = time.monotonic()
        self.stale_wall_s = stale_wall_s
        self.subscription = None
        if use_sim_time:
            if node is None:
                raise ValueError('A spinning ROS node is required for /clock')
            from rosgraph_msgs.msg import Clock
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            self.subscription = node.create_subscription(Clock, '/clock', self._receive,
                QoSProfile(depth=128, reliability=ReliabilityPolicy.RELIABLE))

    def _receive(self, message):
        self.feed(message.clock.sec + message.clock.nanosec/1e9)

    def feed(self, stamp):
        if not math.isfinite(stamp):
            raise ValueError('Nonfinite clock')
        with self.condition:
            if self.value is not None and stamp < self.value-1e-7:
                self.error = RuntimeError('Simulation clock reset during active client/server lifetime')
            self.value, self.received = stamp, time.monotonic()
            self.condition.notify_all()

    def now(self):
        if not self.use_sim_time:
            return time.monotonic()
        with self.condition:
            if self.error:
                raise self.error
            if self.value is None or time.monotonic()-self.received > self.stale_wall_s:
                raise TimeoutError('Missing or stale /clock')
            return self.value

    def wait_ready(self, timeout=30):
        if not self.use_sim_time:
            return self.now()
        deadline = time.monotonic()+timeout
        with self.condition:
            while self.value is None and time.monotonic() < deadline:
                self.condition.wait(min(.05, max(0, deadline-time.monotonic())))
        return self.now()

    def wait_until(self, deadline, stop):
        while not stop.is_set():
            now = self.now()
            if now+1e-8 >= deadline:
                return now
            if self.use_sim_time:
                with self.condition:
                    self.condition.wait(.02)
            else:
                stop.wait(min(.02, deadline-now))
        return None


def install_client(client, use_sim_time, observe_every=10):
    """Change only this episode client's control loop and observation cadence."""
    if observe_every != 10:
        raise ValueError('Phase3 freezes re-observation at ten executed actions')
    clock = PacingClock(use_sim_time, getattr(client.robot, '_node', None))
    clock.wait_ready()
    client.eval_clock = clock
    client.control_loop = MethodType(paced_control_loop, client)


def paced_control_loop(self, task, verbose=False, *, stop_event=None, max_duration_s=None, max_actions=None):
    """Clock-paced actuator worker with one observation worker; no tick-thread I/O."""
    from lerobot.async_inference.robot_client import _configure_dispatch_thread
    clock = self.eval_clock
    self.start_barrier.wait()
    if self._observation_process:
        self._observation_process.resume()
        with self.action_queue_lock, self.latest_action_lock:
            self._observation_process.update_cursor(self.latest_action, self.action_queue.qsize())
    started, wall_started = clock.now(), time.perf_counter()
    finish, observation_due = threading.Event(), threading.Event()
    observation_due.set()
    errors = Queue()
    import gc
    owns_freeze = self.config.dispatch_fifo_priority > 0 and gc.get_freeze_count() == 0
    if owns_freeze:
        gc.collect(); gc.freeze()
    trace, send_costs, waits, observation_events = [], [], [], []
    counts = dict(actions_sent=0, empty_action_ticks=0, empty_action_ticks_after_startup=0,
                  missed_action_ticks=0, missed_action_ticks_after_startup=0,
                  nonmonotonic_action_steps=0, observations_captured=0, observation_retries=0)
    scheduler = {}
    last_step = -1
    request_baseline = [0, 0]

    def cancelled():
        return (finish.is_set() or not self.running or (stop_event is not None and stop_event.is_set())
                or (max_duration_s is not None and clock.now()-started >= max_duration_s))

    def dispatch():
        nonlocal last_step
        empty_streak = 0
        scheduler.update(_configure_dispatch_thread(self.config))
        schedule = TickSchedule(clock.now(), self.config.fps)
        while not cancelled():
            now = clock.wait_until(schedule.deadline, finish)
            if now is None or cancelled():
                break
            expected, missed = schedule.take(now)
            counts['missed_action_ticks'] += missed
            if counts['actions_sent']:
                counts['missed_action_ticks_after_startup'] += missed
            try:
                self.control_loop_action(verbose)
            except Empty:
                counts['empty_action_ticks'] += 1
                empty_streak += 1
                if counts['actions_sent']:
                    counts['empty_action_ticks_after_startup'] += 1
                # An observation/chunk can fail after the last ten-action
                # trigger. Retry at ten empty sim slots, never on every tick.
                if min(empty_streak, counts['empty_action_ticks']-request_baseline[1]) >= 10 and not observation_due.is_set():
                    counts['observation_retries'] += 1
                    self.must_go.set()  # a retry must pass native duplicate-step filtering
                    observation_due.set()
            else:
                empty_streak = 0
                stamp = clock.now()
                step = self.latest_action
                counts['nonmonotonic_action_steps'] += step <= last_step
                last_step = step
                counts['actions_sent'] += 1
                trace.append(dict(clock_s=stamp, deadline_s=expected, jitter_ms=(stamp-expected)*1000,
                                  wall_s=self._last_dispatch_time-wall_started, step=step))
                send_costs.append(self._last_dispatch_cost_s*1000)
                waits.append(self._last_queue_wait_s*1000)
                if counts['actions_sent']-request_baseline[0] >= 10:
                    observation_due.set()
                if max_actions and counts['actions_sent'] >= max_actions:
                    finish.set()

    def observe():
        nonlocal request_baseline
        while not cancelled():
            if observation_due.wait(.02):
                observation_due.clear()
                previous = request_baseline
                request_baseline = [counts['actions_sent'], counts['empty_action_ticks']]
                observation_events.append(dict(clock_s=clock.now(), executed_actions=counts['actions_sent'],
                                               actions_since_request=request_baseline[0]-previous[0],
                                               empty_slots_since_request=request_baseline[1]-previous[1],
                                               next_step=self.latest_action+1, queue_depth=self.action_queue.qsize()))
                # This is the native camera/decode/serialization/RPC implementation.
                self.control_loop_observation(task, verbose, finish)
                counts['observations_captured'] += 1

    def guarded(target):
        try:
            target()
        except Exception as error:
            errors.put(error)
            finish.set()

    workers = [threading.Thread(target=guarded, args=(fn,), name=name) for fn, name in
               ((dispatch, 'robot_action_dispatch'), (observe, 'robot_observations'))]
    try:
        for worker in workers:
            worker.start()
        while not cancelled():
            finish.wait(.02)
    finally:
        try:
            stopped = clock.now()
        except (RuntimeError, TimeoutError):
            stopped = clock.value if clock.value is not None else started
        finish.set()
        if self._observation_process:
            self._observation_process.cancel_pending_capture()
        for worker in workers:
            worker.join()
        if owns_freeze:
            gc.unfreeze()
        span = trace[-1]['clock_s']-trace[0]['clock_s'] if len(trace)>1 else 0
        wall_span = trace[-1]['wall_s']-trace[0]['wall_s'] if len(trace)>1 else 0
        jitter = sorted(abs(t['jitter_ms']) for t in trace)
        self.control_loop_stats = dict(counts, duration_s=stopped-started,
            wall_duration_s=time.perf_counter()-wall_started, target_fps=self.config.fps,
            clock_basis='simulation' if clock.use_sim_time else 'wall',
            dispatch_scheduler=scheduler, observe_every_actions=10,
            observation_cadence_events=observation_events,
            dispatch_times_s=[t['wall_s'] for t in trace], action_steps=[t['step'] for t in trace],
            sim_dispatch_trace=trace if clock.use_sim_time else [],
            clock_dispatch_trace=trace, sim_dispatch_jitter_p95_ms=jitter[math.ceil(.95*len(jitter))-1] if jitter else None,
            effective_action_fps=(len(trace)-1)/span if span else 0,
            effective_wall_action_fps=(len(trace)-1)/wall_span if wall_span else 0,
            send_action_costs_ms=send_costs, queue_waits_ms=waits)
    if not errors.empty():
        raise errors.get()
    return None, None


def clocked_capture_worker(connection, robot_config, cursor, must_go, cancel_capture, server_address, timeout_s):
    """Decorate native isolated capture with a clock stamp in auxiliary metadata."""
    from lerobot.async_inference import observation_process as production
    original_factory = production.make_robot_from_config
    def factory(config):
        robot = original_factory(config)
        connect = robot.connect
        get_observation = robot.get_observation
        holder = {}
        def connected(*args, **kwargs):
            result = connect(*args, **kwargs)
            holder['clock'] = PacingClock(True, robot._node)
            holder['clock'].wait_ready()
            return result
        def observed():
            before = holder['clock'].now()
            raw = get_observation()
            raw['_eval_capture_clock_s'] = before
            return raw
        robot.connect, robot.get_observation = connected, observed
        return robot
    production.make_robot_from_config = factory
    production._capture_worker(connection, robot_config, cursor, must_go, cancel_capture, server_address, timeout_s)


def enable_clocked_capture():
    from lerobot.async_inference import observation_process as production
    # Keep the native worker for the spawn child; install only the spawn target.
    from lerobot.async_inference import robot_client
    class ClockedObservationProcess(production.ObservationProcess):
        def __init__(self, *args, **kwargs):
            original = production._capture_worker
            try:
                production._capture_worker = clocked_capture_worker
                super().__init__(*args, **kwargs)
            finally:
                production._capture_worker = original
    robot_client.ObservationProcess = ClockedObservationProcess


def install_server(use_sim_time):
    """Subclass timing at the native server's model call, keeping normalization/RPC."""
    if not use_sim_time:
        return  # real hardware keeps the native wall-time RTC implementation
    from lerobot.async_inference import policy_server as production
    import os
    backend = os.environ.get('SOARM_EVAL_INFERENCE_MODE', 'eager')
    if backend not in ('eager', 'max-autotune-no-cudagraphs'):
        raise ValueError('Unsupported eval inference mode')
    native_factory = production.get_policy_class
    def policy_factory(name):
        policy_class = native_factory(name)
        class Loader:
            @staticmethod
            def from_pretrained(path):
                from lerobot.configs.policies import PreTrainedConfig
                config = PreTrainedConfig.from_pretrained(path)
                config.compile_model = backend != 'eager'
                if config.compile_model:
                    config.compile_mode = backend
                production.logging.info('EVAL_INFERENCE_MODE=%s checkpoint=%s', backend, path)
                return policy_class.from_pretrained(path, config=config)
        return Loader
    production.get_policy_class = policy_factory
    native = production.PolicyServer
    class ClockedPolicyServer(native):
        def __init__(self, config):
            super().__init__(config)
            self.eval_clock = make_server_clock(use_sim_time)
            self.eval_prediction_s = 0.

        def _reset_server(self):
            result = super()._reset_server()
            self.eval_prediction_s = 0.
            return result

        def _obs_sanity_checks(self, obs, previous_obs):
            # Ten-action re-observation is explicit even for a stationary robot;
            # similarity filtering would also skip the guided compile warmup.
            with self._predicted_timesteps_lock:
                return obs.get_timestep() not in self._predicted_timesteps

        def _predict_action_chunk(self, observation_t):
            before = self.eval_clock.now()
            result = super()._predict_action_chunk(observation_t)
            self.eval_prediction_s = max(0., self.eval_clock.now()-before)
            if result:
                result[0].eval_rtc = dict(getattr(self, 'eval_rtc_trace', {}), measured_prediction_sim_s=self.eval_prediction_s)
            return result

        def _get_action_chunk(self, observation):
            kwargs = {}
            step = self.last_processed_obs.get_timestep()
            rtc = getattr(self.policy.config, 'rtc_config', None)
            previous = self._rtc_previous_chunk
            offset = step-self._rtc_chunk_start
            tail = previous[:, offset:].clone() if previous is not None and 0 <= offset < previous.shape[1] else None
            if rtc is not None and rtc.enabled:
                raw = self.last_processed_obs.get_observation()
                capture = raw['_eval_capture_clock_s'] if use_sim_time else self.eval_clock.now()
                now = self.eval_clock.now()
                delay = delay_steps(self.eval_prediction_s, capture, now, self.config.fps, tail.shape[1]) if tail is not None else 0
                kwargs = dict(prev_chunk_left_over=tail, inference_delay=delay)
                self.eval_rtc_trace = dict(step=step,prefix_steps=0 if tail is None else tail.shape[1],
                    inference_delay=delay,capture_clock_skew_s=capture-now)
                self.logger.info('EVAL_RTC step=%d prefix_steps=%d inference_delay=%d measured_prediction_clock_s=%.6f',
                                 step, 0 if tail is None else tail.shape[1], delay, self.eval_prediction_s)
            chunk = self.policy.predict_action_chunk(observation, **kwargs)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, :self.actions_per_chunk, :]
            self._rtc_previous_chunk, self._rtc_chunk_start = chunk.detach().clone(), step
            return chunk
    production.PolicyServer = ClockedPolicyServer


def make_server_clock(use_sim_time):
    if not use_sim_time:
        return PacingClock(False)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    node = rclpy.create_node('eval_policy_clock')
    clock = PacingClock(True, node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True, name='eval_server_clock').start()
    clock.wait_ready()
    return clock


if __name__ == '__main__':
    import sys
    mode = sys.argv.pop(1)
    if mode not in ('sim', 'wall'):
        raise ValueError('Use sim or wall')
    install_server(mode == 'sim')
    from lerobot.async_inference.policy_server import serve
    serve()
