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
    cfg = draccus.parse(RobotClientConfig, args=sys.argv[1:])
    client = RobotClient(cfg)
    state = {"paused": True, "ep_t0": float("inf"), "chunks": 0, "first_chunk_t": None}

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
            if state["first_chunk_t"] is None:
                state["first_chunk_t"] = time.time() - state["ep_t0"]
            orig_aggregate(incoming, aggregate_fn)

    client._aggregate_action_queues = gated_aggregate

    t = time.perf_counter()
    if not client.start():  # Ready + SendPolicyInstructions (server loads policy)
        emit("FATAL", {"msg": "client.start() failed"})
        return 1
    load_s = time.perf_counter() - t
    client.start_barrier = threading.Barrier(
        1
    )  # receiver no longer waits for control_loop
    threading.Thread(target=client.receive_actions, daemon=True).start()
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

    while True:
        line = cmds.get()
        if line == "QUIT":
            break
        parts = line.split()
        if parts[0] != "RUN":
            continue
        ep, max_s = parts[1], float(parts[2])
        max_actions = int(parts[3]) if len(parts) > 3 else 10**9
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
            state.update(chunks=0, first_chunk_t=None, ep_t0=time.time(), paused=False)
        emit("EPSTART", {"ep": ep, "t0": state["ep_t0"]})
        n_actions, deadline = 0, time.perf_counter() + max_s
        while (
            time.perf_counter() < deadline
            and n_actions < max_actions
            and not abort.is_set()
        ):
            t_loop = time.perf_counter()
            if client.actions_available():
                client.control_loop_action(False)
                n_actions += 1
            if client._ready_to_send_observation():
                client.control_loop_observation(cfg.task, False)
            time.sleep(max(0, cfg.environment_dt - (time.perf_counter() - t_loop)))
        with state_lock:
            state["paused"] = True
            dur = time.time() - state["ep_t0"]
            state["ep_t0"] = float("inf")
        with client.action_queue_lock:
            client.action_queue = Queue()
        # Wait for in-flight inference and clear RTC/model queues before homing.
        client.stub.Ready(services_pb2.Empty())
        emit(
            "EPDONE",
            {
                "ep": ep,
                "duration_s": round(dur, 2),
                "actions_sent": n_actions,
                "chunks": state["chunks"],
                "first_chunk_s": state["first_chunk_t"],
                "aborted": abort.is_set(),
                "hit_action_budget": n_actions >= max_actions,
            },
        )

    client.stop()
    emit("BYE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
