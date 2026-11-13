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

For paired evaluation, pass `--condition NAME --seeds 900000,900001` (one
unique seed per episode), with the same `--scene-bank /persistent/path` across
conditions. `scene_reset.py` runs through `run_python`, reuses the extension's
blue-visible sampler, saves exact physical block/cup poses, and restores those
poses with zero velocities before subsequent uses of a seed. It neither edits
the extension nor adds/removes scene bodies. The bank's baseline must come from
the intended evaluation scene. The arm is homed through the normal FJT path.
Kit's frame cap is reapplied/read back after each reset; `--sim-rate-hz` defaults
to 17.4. A mismatch is refused. Use `--fps 19.4 --sim-rate-hz 19.4` for matched
camera/client cadence. These rec_222812 models keep `/joint_states` feedback.

The CSV includes condition, seed, RTC setting, requested/effective action Hz,
sim cap read-back and onset times of sustained grasp/place. Outcomes are
`success`, `grasped-not-placed`, and `no-grasp`. A grasp requires one particular
blue block to sustain its lift; success also requires a sustained lift.
Per-episode JSON additionally retains physical scene receipts, GPU snapshots,
CPU load, all dispatch measurements and resolved sim timing settings.

Request/arrival telemetry runs in observation and receiver workers, with no
dispatch hook or per-tick I/O. Each `chunks.csv` matches observation timestamps
to capture-request queue depth and request-to-chunk-arrival latency (capture,
transport, server queueing and inference combined). CSV summaries include
first-chunk delay, empty post-startup ticks/count percentage and fixed refill
settings. Empty queues are measured outcomes; scheduling validity is evaluated
separately from effective Hz. The scoring rule requires zero missed dispatch
slots, no stale-camera aborts, and bounded send costs. Earlier untraced runs
are diagnostics. The refill threshold remains 0.5 with 50 actions per chunk.

`--dataset-brightness` is an opt-in photometric diagnostic for the isolated
evaluation sensor process. Both raw RGB camera arrays are divided by
`[1.12162, 1.11828, 1.11650]`, clipped, rounded and converted back to uint8 before
the existing observation RPC. State/task, viewport images and dispatch are
unchanged. Default is off; each result records the enabled flag, divisors and
camera names. The first actual sensor observation also returns raw/corrected
image hashes and ROI medians as a child-process wiring proof, written at episode
end. The top ROI is the calibration table patch; the wrist ROI is only a fixed
image patch. This harness wrapper does not modify the LeRobot fork.

The 1.10a inverse-gain test reused the ten 1.9 NEW/17.4/RTC-off seeds: treatment
and historical baseline both had 0/10 placements and no sustained object lift.
Mean paired closest command-FK distance changed by +11.71 mm (farther), with
two seeds >=1 mm closer and seven >=1 mm farther. This small, non-interleaved
comparison with unpaired policy noise does not establish a causal lighting
effect. Treatment had zero scheduling misses/stale-camera aborts. The Mac
`phase1/1.10-decision/lighting-ab/` folder contains the result-first README,
paired CSV, first-child calibration proof and short clips for three seeds.
After running `characterize_failures.py` on a report with `raw/baseline/` and
`raw/treatment/`, generate this comparison with
`python3 linux-env/scripts/eval/report_lighting_ab.py /path/to/lighting-ab`.

The orchestration process records clock/wrist/workspace message arrival stamps.
An in-memory Kit update probe records wall/sim time without touching extension
files, and per-episode Isaac major-fault counts test page-in stalls. This allows
a camera timeout to be compared against app-frame and topic gaps; the 500 ms
camera deadline is unchanged. Samples are written at episode end.

`--fresh-client-per-episode` is a diagnostic A/B option: the policy server stays
alive while each episode gets a new client/sensor process (additional logs are
under `client-NN/`). Episode client receipts include goal-handle/pending-future
counts, executor tasks, status-array sizes and thread/process counts before/after
each episode. For action-based clients, status sizes are observed at the existing
ActionClient doorway. The streaming client has no ActionClient/status subscription;
the separate orchestration process monitors controller status sizes so only the
discrete homing goals should add retained history.

`--model PRESET` resolves the same definition as the GUI and `deploy_sim.sh`.
The shared async launcher applies the preset's camera-key mapping even when
the GUI/deployer passes its resolved checkpoint path; explicit `--cameras-json`
takes precedence. `--pretrained` still accepts local directories and HF IDs.

To reproduce the report on the Mac, copy each condition beneath `raw/` and
the shared bank into the report directory, then run
`python3 linux-env/scripts/eval/report_results.py /path/to/report [--complete]`.
The report generator needs matplotlib, Pillow and ffmpeg. It writes the combined
CSV, Wilson-interval chart, paired differences/exact McNemar probabilities,
video contact sheet and README. `--complete` refuses a partial 50-episode matrix.

For post-hoc approach/closure/lift characterization, first run
`python3 linux-env/scripts/eval/characterize_failures.py /path/to/report --fk-module
vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py`.
This reuses production URDF FK on recorded controller-reference states; it does
not recover historical measured TCP poses. Sidecars and `failure-characterization.csv`
retain closest blue-centre distance, closing motion, minimum opening/position,
and sustained lifts of any colour. The report merges these fields into its CSV.
The linear jaw-gap estimate snapshots the production coefficients 0.019 m and
0.0749 m/rad; it does not establish finger contact or a completed grasp.

The completed 1.9 run has 50 fully traced episodes: OLD 0/5 before and 0/5 after;
NEW 0/10 in each of four RTC/rate conditions. All scored episodes have zero
scheduling misses and no stale-camera abort. Success intervals remain wide.
The Mac report includes all 50 verified videos, paired timing/starvation metrics,
four-frame NEW sequences and rec_222812-vs-live home-camera comparisons under
`coursework-eee598/roadmap-2026-11/phase1/1.9-eval/`.

`results.csv` and `results.jsonl` contain episode ID, outcome, elapsed policy wall time, dispatched actions, received chunks, measured effective action fps, maximum blue lift, failure note and video path. `effective_action_fps` measures the first-to-last dispatch interval, excluding initial chunk wait. Each `epNN/` contains `viewport.mp4`, pose samples and a detailed result JSON. `server.log`/`client.log` record inference and reset behavior. `run_config.json` captures resolved settings, GPU UUID and repository/fork commits; each episode also reports nonmonotonic action-index count. The viewport bridge encodes at nominal 30 fps; the harness rescales MP4 packet timestamps to measured recording wall time, so slow simulator rendering does not speed up playback.

Success requires a blue lego centre inside `drop_1` continuously for two wall seconds and still inside when the client acknowledges its stop. The check transforms `/objects_poses_sim` coordinates into the cup-local frame from `/drop_poses`, including cup orientation. It uses a conservative interior radius of 0.033 m (measured outer radius ≈0.039 m) and height 0.0965 m with floor/rim clearance. This is a geometric centre test, not a contact or full-object containment sensor. `grasped-not-placed` means a blue lego centre rose ≥0.024 m from its randomized initial height for at least 0.5 s, without a final success; it is a sustained-lift proxy for grasping. `no-grasp` means neither condition occurred. Missing/stale pose streams or unavailable actions fail visibly; they never become success. Timeout/failure notes distinguish the cause.

The client `RUN`/`ABORT`/`QUIT` protocol is implemented by `episode_client.py`. While paused, it sends no observations or actions and rejects incoming chunks by timestamp. A production `Ready` reset waits for in-flight inference and clears RTC/model state without reloading weights. The wrapper uses the shared inference environment entrance: orchestration, scoring and policy processes all run in pixi-Jazzy, while the existing controllers remain on system Humble. Scoring callbacks run on one executor thread, separately from orchestration. Both the stock client and episode client use the same absolute-deadline action dispatcher, with observation requests on a second thread. Deployment additionally spawns an observation-only robot process: camera reads, RGB decoding, serialization and the full observation RPC stay outside the dispatch process's GIL. Only small capture metadata returns. Atomic nonblocking cursor updates preserve capture-start action indices for RTC. Cancellation joins both threads and suppresses a capture pending at stop; the persistent sensor process closes when the client exits.

Deployment defaults to `--observation-process --dispatch-priority 10`. Only the dispatch native thread receives Linux SCHED_FIFO priority 10; the main, receiver and capture process remain normal priority. This requires permission for `os.sched_setscheduler` or the available `sudo -n chrt -f -p 10 <owned-tid>` path. A permission failure stops visibly. No system settings are changed. Override `--dispatch-priority 0` for best-effort timing or `--no-dispatch-sudo` to forbid escalation; `--no-observation-process` retains the independent-thread capture mode. `deploy_sim.sh` accepts the same flags and `SOARM_DISPATCH_PRIORITY` defaults to 10.

Before each priority-enabled episode, the client collects garbage and freezes its stable imported Torch/ROS baseline so full cyclic collections do not hold the dispatch GIL for hundreds of milliseconds. New cycles still collect during the run. After the workers join, it unfreezes a baseline it owns; a pre-existing caller-owned permanent generation remains frozen and is extended. The client reports GC pauses, actual actuator-call intervals, queue waits, missed slots and scheduler settings. It skips overdue slots without catch-up bursts, and never repeats an action to hide an empty queue.

## RTC replay

See `replay_rtc.py` and [rtc-report.md](rtc-report.md) for the paired dataset experiment, its plot and precise metric. The replay invokes the actual `PolicyServer` prediction pipeline directly without network transport, ROS or Isaac. It needs the LeRobot dataset extra, PyAV and matplotlib in the existing pixi environment. It uses the same trained checkpoint, processor chain and normalized previous-chunk guidance as deployment.


## Verified two-episode smoke (2026-11-13)

The coordinated `smoke-verified` run completed two 45-second policy windows,
with the checkpoint loaded once and an empty RTC prefix at each episode start.
Both viewport MP4s decode fully and their durations match recording wall time
within 0.2 seconds. Both episodes returned home, randomized six blocks with
at least one blue visible, ran policy actions, paused/flushed, and saved results.

| Episode | Outcome | Policy time | Actions | Chunks | Effective action Hz | Repeated/reversed steps |
|---|---|---:|---:|---:|---:|---:|
| ep00 | fail: no sustained blue lift | 45.03 s | 526 | 38 | 12.05 | 0 |
| ep01 | fail: no sustained blue lift | 45.08 s | 460 | 31 | 10.33 | 0 |

The requested fps was 17.4 and FJT duration 0.057471 s. Actual rates were lower
on the loaded host; these observations establish harness operation, not task
success or a full evaluation. The initial attempt was stopped by the freshness
guard after the system-Humble monitor stopped receiving all three pose/state
streams for two seconds, while the Jazzy robot client continued. The verified
entry uses the same Jazzy runtime for monitoring and policy clients. Shutdown
also waits for the camera spin thread before destroying ROS handles, and joins
the action receiver; the verified run exited cleanly. The initial attempt is
retained under `~/w_policy_phase1/smoke/`, separately from verified results.

The Mac deliverables live in `coursework-eee598/roadmap-2026-11/phase1/1.3-1.8-policy/`:
`smoke/results.csv`, `smoke/results.jsonl`, `smoke/ep00/viewport.mp4`,
`smoke/ep01/viewport.mp4` and `smoke/verification.json`. `run_config.json` records
HEAD at run start; `source.patch` and `fork-source.patch` capture the pending
runtime/shutdown fixes used in that run. Final commit IDs are in the deliverable
README. Isaac and the control stack were left running; only the harness-owned
policy processes were stopped. Item 1.9 was not run.

## Verified independent dispatcher (ROADMAP 1.15)

The coordinated 45-second verification on GPU 1 reached **17.3999 Hz**:
764 distinct actions, 53 chunks, zero missed slots, zero empty slots after
startup and zero repeated/reversed steps. The initial chunk wait caused
19 empty startup ticks. p95 absolute interval jitter was 0.677 ms; maximum
was 25.530 ms. Load at the end was 16.39/16.06/15.98 on 28 logical CPUs.
Only the dispatcher used FIFO10; main-thread policy remained normal.
The episode timed out with no sustained blue lift; this verifies timing.

The final offline suite passes 52 async tests, including 40/80/160 ms blocking
camera reads at 17.4 Hz, a CPU-bound spawned sensor, child-owned gRPC, queue
races, cancellation during capture/serialization, worker joins and GC
ownership on actuator error. The full live MP4 decodes and lasts 47.105 s
versus 47.142 s recording wall time. Own policy processes exited; Isaac
94191 and control_gui 79380 were left running. Raw traces, failed diagnostic
attempts, CSV, plots, review and video are in
`~/w_policy_phase1/1.15/`, copied to the Mac
`coursework-eee598/roadmap-2026-11/phase1/1.3-1.8-policy/1.15/`.
