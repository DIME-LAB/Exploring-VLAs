# Policy timing, RTC replay and episode harness

This directory contains an interim, repeatable evaluation path for the deployed SmolVLA policy. It uses the same configuration builder and `RobotClient` as `run_async_inference_sim.py`, keeps model/server/client processes alive between episodes, and flushes client actions plus server RTC/model queues at episode boundaries. It homes with FollowJointTrajectory to the recorded initial pose, randomizes with at least one blue lego visible, runs until sustained blue-in-cup success or timeout, pauses and flushes the policy, saves viewport video and logs every result in CSV/JSONL. Isaac must already be running; this harness never starts or stops it.

## Run

Coordinate exclusive scene/arm use first. The existing ROS controllers must be active and `/objects_poses_sim`, `/drop_poses` and `/joint_states` must publish fresh messages. The FJT homing path does not require control_gui or an X display. The randomization and recording tools need the existing bridge on port 8767 (override `SOARM_KIT_MCP_PORT`). No other policy client may command the arm.

```bash
# Run from Exploring-VLAs on the Linux host; GPU 1 is pinned by UUID.
bash linux-env/scripts/eval/run_episodes.sh \
  --episodes 2 --max-s 45 --out "$HOME/w_policy_phase1/smoke" \
  --server-port 8081

# Compare with RTC disabled. Choose a new output directory for every run.
bash linux-env/scripts/eval/run_episodes.sh \
  --episodes 2 --max-s 45 --out "$HOME/w_policy_phase1/smoke-no-rtc" \
  --server-port 8081 --no-rtc

# Same timing knobs on the normal wrapper and deployment entrance:
bash linux-env/scripts/inference_async_lerobot.sh --dry-run --fps 17.4
bash linux-env/scripts/deploy_sim.sh up --fps 17.4 --action-duration-s .057471 --no-gui
```

The runner accepts deployment launcher flags, including `--pretrained`, `--task`, `--fps`, `--action-duration-s`, `--actions-per-chunk`, `--chunk-size-threshold`, `--aggregate-fn`, `--no-rtc` and `--rtc.*`. Fractional fps defaults to the single `RECORDED_FPS=17.4` constant in the launcher, selected from the recording-rate audit’s 17.43 Hz motion-phase estimate; 19.4 remains selectable. Trajectory duration defaults to `1/fps`. Both camera topics default to `rgb8`, matching ROS2CameraHelper output. The runner starts its own server on the requested free port and refuses an occupied port; it stops only its own processes on exit. The checkpoint loads once for the entire run, rather than once per episode.

## Outputs and scoring

`results.csv` and `results.jsonl` contain episode ID, outcome, elapsed policy wall time, dispatched actions, received chunks, maximum blue lift, failure note and video path. Each `epNN/` contains `viewport.mp4`, pose samples and a detailed result JSON. `server.log`/`client.log` record inference and reset behavior. The viewport bridge encodes at nominal 30 fps; the harness rescales MP4 packet timestamps to measured recording wall time, so slow simulator rendering does not speed up playback.

Success requires a blue lego centre inside `drop_1` continuously for two wall seconds and still inside when the client acknowledges its stop. The check transforms `/objects_poses_sim` coordinates into the cup-local frame from `/drop_poses`, including cup orientation. It uses a conservative interior radius of 0.033 m (measured outer radius ≈0.039 m) and height 0.0965 m with floor/rim clearance. This is a geometric centre test, not a contact or full-object containment sensor. `grasped-only` means a blue lego centre rose ≥0.024 m from its randomized initial height for at least 0.5 s, without a final success; it is a sustained-lift proxy for grasping. `fail` means neither condition occurred. Missing/stale pose streams or unavailable actions fail visibly; they never become success. Timeout/failure notes distinguish the cause.

The client `RUN`/`ABORT`/`QUIT` protocol is implemented by `episode_client.py`. While paused, it sends no observations or actions and rejects incoming chunks by timestamp. A production `Ready` reset waits for in-flight inference and clears RTC/model state without reloading weights. ROS callbacks run separately from orchestration. The stock robot action/observation loop still shares one thread: camera blocking can stretch actual action ticks below requested fps.

## RTC replay

See `replay_rtc.py` and [rtc-report.md](rtc-report.md) for the paired dataset experiment, its plot and precise metric. The replay invokes the actual `PolicyServer` prediction pipeline directly without network transport, ROS or Isaac. It needs the LeRobot dataset extra, PyAV and matplotlib in the existing pixi environment. It uses the same trained checkpoint, processor chain and normalized previous-chunk guidance as deployment.
