#!/usr/bin/env python3
# Reference: lerobot/src/lerobot/async_inference/robot_client.py (control_loop / async_client)
"""Episode-gated LeRobot async robot client for closed-loop eval.

Same RobotClient + same CLI config as run_async_inference_sim.py's client,
but one long-lived process (policy loaded ONCE on the server) whose control
loop only runs between RUN and the episode deadline. Between episodes it is
paused: no observations sent, no actions dispatched.

stdin protocol (one command per line):
  RUN <episode_id> <max_seconds> [max_actions] -> runs one episode, prints "EPDONE <json>"
  ABORT                           -> ends the running episode early
  QUIT                            -> disconnect and exit
stdout lines starting with "READY", "EPSTART", "EPDONE" are for the orchestrator.
"""

from __future__ import annotations

import json
import argparse
import os
import sys
import threading
import time
from queue import Queue

import draccus
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2


def emit(tag: str, payload: dict | None = None) -> None:
    sys.stdout.write(f"{tag} {json.dumps(payload or {})}\n")
    sys.stdout.flush()


def main() -> int:
    diagnostic_parser = argparse.ArgumentParser(add_help=False)
    diagnostic_parser.add_argument("--dataset-brightness", action="store_true")
    diagnostic, remaining = diagnostic_parser.parse_known_args(sys.argv[1:])
    cfg = draccus.parse(RobotClientConfig, args=remaining)
    from observation_transform import enable_isolated_correction, transform_metadata
    correction = transform_metadata(diagnostic.dataset_brightness)
    if diagnostic.dataset_brightness:
        if not cfg.observation_process:
            raise ValueError("--dataset-brightness requires isolated observation capture")
        enable_isolated_correction()
    client = RobotClient(cfg)
    status_arrays = {}
    for label in ("arm", "gripper"):
        action_client = getattr(client.robot, f"_{label}_action_client", None)
        if action_client is None:
            continue
        status_arrays[label] = {"messages": 0, "latest_size": 0, "max_size": 0}
        original = action_client.take_data
        def counted_take(original=original, label=label):
            data = original()
            if "status" in data:
                size = len(data["status"].status_list)
                stats = status_arrays[label]
                stats.update(messages=stats["messages"]+1, latest_size=size,
                             max_size=max(size, stats["max_size"]))
            return data
        action_client.take_data = counted_take

    def resources():
        node = client.robot._node
        executor = node.executor
        action_clients = {}
        for label in ("arm", "gripper"):
            action_client = getattr(client.robot, f"_{label}_action_client", None)
            if action_client is None:
                continue
            action_clients[label] = {attr: len(getattr(action_client, attr, {})) for attr in (
                "_goal_handles", "_pending_goal_requests", "_pending_result_requests",
                "_pending_cancel_requests", "_feedback_callbacks", "_futures")}
            action_clients[label]["live_goal_handles"] = sum(ref() is not None for ref in list(action_client._goal_handles.values()))
        return dict(pid=os.getpid(), python_threads=[t.name for t in threading.enumerate()],
                    native_threads=len(os.listdir('/proc/self/task')),
                    observation_pid=client._observation_process.process.pid if client._observation_process else None,
                    executor_tasks=len(getattr(executor, '_tasks', [])),
                    action_clients=action_clients,
                    status_arrays={k: dict(v) for k,v in status_arrays.items()})
    state = {"paused": True, "ep_t0": float("inf"), "chunks": 0, "first_chunk_t": None,
             "request_events": [], "chunk_events": []}
    request_context = {}

    # Telemetry runs only in the observation/receiver workers. Dispatch is untouched.
    original_observation = client.control_loop_observation
    def traced_observation(*args, **kwargs):
        request_context.clear()
        request_context.update(trigger_wall_time=time.time(),
                               queue_depth_at_capture_request=client.action_queue.qsize(),
                               chunk_size=client.action_chunk_size,
                               chunk_size_threshold=cfg.chunk_size_threshold)
        return original_observation(*args, **kwargs)
    client.control_loop_observation = traced_observation

    if client._observation_process:
        original_capture = client._observation_process.capture
        def traced_capture(task):
            try:
                captured = original_capture(task)
            except Exception as error:
                state["request_events"].append(dict(request_context, error=str(error), error_wall_time=time.time()))
                raise
            if captured.sent and not captured.cancelled:
                proof = getattr(captured, "observation_transform_proof", None)
                if proof:
                    state["observation_transform_proof"] = proof
                state["request_events"].append(dict(request_context,
                    observation_timestamp=captured.timestamp, timestep=captured.timestep,
                    capture_rpc_s=captured.capture_s, capture_return_wall_time=time.time()))
            return captured
        client._observation_process.capture = traced_capture
    else:
        original_send = client.send_observation
        def traced_send(observation):
            state["request_events"].append(dict(request_context,
                observation_timestamp=observation.get_timestamp(), timestep=observation.get_timestep(),
                rpc_start_wall_time=time.time()))
            return original_send(observation)
        client.send_observation = traced_send

    state_lock = threading.Lock()

    # Drop chunks that belong to a previous episode (or arrive while paused):
    # TimedAction timestamps derive from the observation timestamp.
    orig_aggregate = client._aggregate_action_queues

    def gated_aggregate(incoming, aggregate_fn=None):
        with state_lock:
            # Only the FIRST action carries the observation timestamp. Later
            # action timestamps extend into the future, even for stale chunks.
            if (
                state["paused"]
                or not incoming
                or incoming[0].get_timestamp() < state["ep_t0"]
            ):
                return
            state["chunks"] += 1
            arrival = time.time()
            state["chunk_events"].append(dict(
                arrival_wall_time=arrival, observation_timestamp=incoming[0].get_timestamp(),
                first_step=incoming[0].get_timestep(), actions=len(incoming),
                observation_to_arrival_s=arrival-incoming[0].get_timestamp(),
                queue_depth_before_merge=client.action_queue.qsize()))
            if state["first_chunk_t"] is None:
                state["first_chunk_t"] = time.time() - state["ep_t0"]
            orig_aggregate(incoming, aggregate_fn)

    client._aggregate_action_queues = gated_aggregate

    t = time.perf_counter()
    if not client.start():  # Ready + SendPolicyInstructions (server loads policy)
        emit("FATAL", {"msg": "client.start() failed"})
        client.stop()
        return 1
    load_s = time.perf_counter() - t
    client.start_barrier = threading.Barrier(
        1
    )  # receiver no longer waits for control_loop
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    emit("READY", {"policy_instructions_s": round(load_s, 2)})

    abort = threading.Event()
    cmds: Queue = Queue()

    def stdin_reader():
        for line in sys.stdin:
            line = line.strip()
            if line == "ABORT":
                abort.set()
            elif line:
                cmds.put(line)
        cmds.put("QUIT")

    threading.Thread(target=stdin_reader, daemon=True).start()

    try:
        while True:
            line = cmds.get()
            if line == "QUIT":
                break
            parts = line.split()
            if parts[0] != "RUN":
                continue
            ep, max_s = parts[1], float(parts[2])
            max_actions = int(parts[3]) if len(parts) > 3 else 10**9
            resources_before = resources()
            abort.clear()
            # ---- per-episode reset of client + server state (no policy reload) ----
            with client.action_queue_lock:
                client.action_queue = Queue()
            with client.latest_action_lock:
                client.latest_action = -1
            client.action_chunk_size = -1
            client.must_go.set()
            client.stub.Ready(
                services_pb2.Empty()
            )  # flush server obs queue + predicted timesteps
            with state_lock:
                state.update(
                    chunks=0, first_chunk_t=None, ep_t0=time.time(), paused=False,
                    request_events=[], chunk_events=[]
                )
                state["observation_transform_proof"] = None
            emit("EPSTART", {"ep": ep, "t0": state["ep_t0"]})
            try:
                client.control_loop(
                    task=cfg.task,
                    stop_event=abort,
                    max_duration_s=max_s,
                    max_actions=max_actions,
                )
            except Exception as error:
                emit("FATAL", {"error": str(error),
                               "metrics": getattr(client, "control_loop_stats", {}),
                               "resources_before": resources_before,
                               "resources_after": resources(),
                               "request_events": state["request_events"],
                               "chunk_events": state["chunk_events"]})
                raise
            metrics = client.control_loop_stats
            with state_lock:
                state["paused"] = True
                dur = metrics["duration_s"]
                state["ep_t0"] = float("inf")
            with client.action_queue_lock:
                client.action_queue = Queue()
            # Wait for in-flight inference and clear RTC/model queues before homing.
            client.stub.Ready(services_pb2.Empty())
            emit(
                "EPDONE",
                {
                    "ep": ep,
                    "observation_transform": correction,
                    "observation_transform_proof": state.get("observation_transform_proof"),
                    "duration_s": round(dur, 2),
                    **metrics,
                    "resources_before": resources_before,
                    "resources_after": resources(),
                    "request_events": state["request_events"],
                    "chunk_events": state["chunk_events"],
                    "chunks": state["chunks"],
                    "first_chunk_s": state["first_chunk_t"],
                    "aborted": abort.is_set(),
                    "hit_action_budget": metrics["actions_sent"] >= max_actions,
                },
            )

    finally:
        client.stop()
        receiver.join(timeout=5)
    emit("BYE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
