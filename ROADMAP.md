# Roadmap — SO-ARM101 SmolVLA in Isaac Sim 6.1

Work happens on `dev`; `main` gets a merge only when a phase's merge gate is met.
Status at `30c6ba1` (main): platform ported to Isaac Sim 6.1 + omni-kit-mcp; scripted control_gui
pick-and-place verified; SmolVLA runs closed-loop but rarely grasps.

## Phase 1 — Deployment fixes (goal: see real policy results before deciding on retraining)

Why: three deployment-side mismatches plausibly explain the missed grasps — action replay 1.55× too
fast (client 30 fps vs the ~19.4 Hz the data was recorded at), RTC never actually applied, and camera
cadence (~14 Hz) not matching the camera-bound recording loop.

- [ ] 1.1 Confirm the recording rate of `rec_222812` from the data itself (count duplicate consecutive
      frames in the dataset videos) — settles 19.4 Hz vs ~0.4× RTF ambiguity.
      Audit (100 episodes, all frames): wall estimate 19.35 Hz (OLS 95% CI 16.67–23.06); calibrated final-home motion 17.43 Hz; exact decoded repeats wrist 24.00%, top 29.87%; historical sim RTF remains unresolved (conditional 0.323× vs 0.419×), so 1.1 stays open. Mac evidence: `coursework-eee598/roadmap-2026-11/phase1/1.1-recording-rate/`.
- [x] 1.2 Pin sim speed: Kit run-loop rate limit (`/app/runLoops/main/rateLimitEnabled`,
      `rateLimitFrequency`) set at launch after the World exists and read back; launcher flag
      `--rate-hz` (default from 1.1). Verify RTF ≈ 0.29 ± 0.02 (17.4 Hz default from 1.1; report load/headroom misses).
- [x] 1.3 Robot client loop at the selected recording rate (fps = 17.4, `action_duration_s` to match); confirm
      dataset `fps` vs real rate handling in async inference.
- [x] 1.4 Camera cadence: one frame per sim frame with sim-time stamps — wrist camera via render
      product + ROS2CameraHelper (pixel parity already measured: 49–50 dB, at renderer noise);
      workspace camera publisher stamps + one publish per frame. Re-verify parity vs dataset frames.
- [x] 1.5 Per-frame CPU waste: fix the pose graphs' invalid TF targets (~88 PoseTree warnings/s);
      make the extension log pane append-only (no 500-label rebuilds); rate-limit prints inside physics
      callbacks (a broken handle once dropped RTF to 0.12).
- [x] 1.6 Invisible robot on cold launch (`cannot find protoPath: /__Prototype_*`): de-instance the
      robot's visual meshes at scene build; verify 3 cold launches.
- [x] 1.7 Real RTC: policy server tracks the leftover chunk + measured inference delay and passes
      `prev_chunk_left_over` / `inference_delay` to `predict_action_chunk` (fork, branch `dev`);
      prove guidance is non-zero in logs.
- [x] 1.8 Episode harness (interim, before RunTask): home → randomize (blue visible) → start policy →
      success / timeout → stop → record viewport mp4 → score from `/objects_poses_sim`.
- [ ] 1.9 Evaluation (restructured 2026-11-13; needs 1.11–1.15 and the 3.0b checkpoint), idle box:
      (a) old checkpoint, 5 episodes before vs after the Phase 1 fixes. It was trained from scratch (see Phase 3),
          so this only confirms whether deployment alone moves it.
      (b) 3.0b fine-tuned checkpoint (main subject), 10 episodes each: RTC on vs off, at 17.4 vs 19.4 Hz.
      Report success, grasp-only, failure mode, effective action Hz, and per-episode videos.
- [ ] 1.10 **Retraining decision** from 1.9 (see Phase 3 criteria). Record results in the coursework README.
- [x] 1.11 GUI + model selection: add every new checkpoint (e.g. the 3.0b `base-ft`) to control_gui's
      Inference dropdown (`INF_MODEL_PRESETS`) and `deploy_sim.sh --model`; make the Inference tab use the
      Phase 1 defaults (17.4 Hz fps, rgb8 cameras, real-RTC on/off toggle); fix the Method radio SEGV and
      wire recording in async mode (both from the April 29 session's open priorities).
- [x] 1.12 Record Sim tab works with the Phase 1 changes (rgb8 cameras, rate limit, fork `dev`); verified by
      recording 2 episodes and loading them with LeRobotDataset (needed before 3.1 data collection).
- [x] 1.13 `soarm101.new_stage` crashes Isaac 6.1 with live ROS action graphs: tear graphs/publishers down
      first or refuse with a clear error.
- [x] 1.14 README: warn that saving files under `isaac/exts/` hot-reloads the extension in a running Isaac
      (stops publishers); edit only with the sim down. Fix `_apply_sim_rate` docstring (physics is 120 Hz;
      60 Hz is the sim-frame/clock rate).
- [x] 1.15 (moved from 2.5c) Decouple observation capture from action dispatch in the robot client.
      Persistent sensor/RPC process, absolute-deadline dispatcher at FIFO10, nonblocking cursor and GC baseline.
      52 async tests; slow-camera dispatch 17.4 Hz. Live 45 s: 17.3999 Hz, zero missed/repeated or
      empty-after-startup slots; load 16.39 on 28 CPUs. GPU1; video/CSV/report under `1.3-1.8-policy/1.15/`.
      Defaults require scheduler permission or noninteractive `sudo -n chrt`; priority 0 remains selectable.
- **Merge gate → main:** 1.1–1.15 done, `deploy_sim.sh up` still one command, README updated.

## Phase 2 — Own LeRobot ROS 2 packages (architecture; review verdict: build our own from the fork's code)

- [ ] 2.1 Shared ROS context module: init only if rclpy not running, own-what-you-start shutdown,
      ref-counted executor, attach to an external context/executor.
- [ ] 2.2 Camera plugin: fresh-frame `async_read` (keeps training timing) + `read_latest(max_age_ms)`,
      header stamps, configurable QoS, row padding/byte order, mono→3ch, per-instance subscriptions,
      Condition instead of polling, CompressedImage. (Check PyPI name; borrow cv_bridge fallback from
      fiveages-sim, Apache-2.0, with attribution; credit huggingface/lerobot PR #866 for the original camera.)
- [ ] 2.3 `lerobot_robot_ros2_joint`: controller groups (action + joints) from config, name map,
      limits from config/URDF, units, `actuate`, `max_relative_target`, `state_max_age_ms`, merged
      JointState cache, goal tracking; SO-101 preset.
- [ ] 2.4 `lerobot_teleoperator_ros2_joint` (commanded-joint topic) for recording; reconcile its
      `action_stale_ms` check with control_gui publishing commands only on change.
- [ ] 2.5 RunTask ROS 2 node: action {instruction, max_duration} → home → run → success/timeout/cancel
      → home; Isaac panel button `soarm101.run_task`; live prompt per goal.
- [ ] 2.5b Prompt control: change the instruction live (ROS topic + Isaac panel text box) and pick the model
      from the Isaac panel; both go through the same RunTask/goal interface.
- [ ] 2.6 Unit tests with fake rclpy messages; CI.
- [ ] 2.7 Switch Exploring-VLAs to PyPI LeRobot + our packages; drop the fork patches; archive the fork
      (never delete — it holds `f26b0a6`).
- [ ] 2.8 Port 1.7 (RTC) upstream-compatible or as a small upstream PR.
- **Merge gate → main:** recording + async inference work on PyPI LeRobot without the fork; eval rerun
  through RunTask matches Phase 1 numbers.

## Phase 3 — Retraining (only if 1.10 says so)

Resolved (2026-11-13, 3.0 weight comparison): the sim checkpoints `smolvla_sim_100ep_fft__10ksteps_h200` and
`..._75S_25R_...` were trained from RANDOM INITIALIZATION. Median per-tensor cosine vs both `lerobot/smolvla_base` and
SmolVLM2 ≈ 0 in every group (vision, text, connector, action expert); all 500 tensor shapes match the base. The
report's "fine-tune the SmolVLA base model" does not hold for these checkpoints. 3.0b (fine-tune FROM smolvla_base)
is therefore a genuinely different run, not a reproduction.

- [x] 3.0 Audit every coursework checkpoint's `train_config.json` (pretrained_path, load_vlm_weights, steps,
      dataset) AND compare weights (per-tensor cosine/rel-L2 vs smolvla_base and SmolVLM2) to decide from-scratch
      vs fine-tuned; record the verdict in the coursework README.
- [ ] 3.0b Fine-tune from `lerobot/smolvla_base` (`--policy.path=lerobot/smolvla_base`) on the existing
      `soarm101_blue_lego_pick_v1` data first (cheapest test of the from-scratch hypothesis), then evaluate
      with the Phase 1 harness.

Decide by: grasp success after Phase 1 (if still ≲ 20%, the model is the bottleneck) and whether
colour/instruction following is required (single task string → language cannot be learned).
Data inventory (2026-11-13, HF + local disks): SIM = only 100 episodes (`soarm101_blue_lego_pick_v1` = `rec_222812`,
blue task); REAL = 440 blue (`arjunsinghyadav2/blue_sort_black_bg_colored_cups_v1_440ep`) + ~300 multi-colour
(`arjunsinghyadav2/lego_sort_300ep_03042026`). `anirudhrani/{0real_100sim,25real_75sim,75real_25sim,50real_50sim}` look
mislabeled (0real_100sim has the real-only set's exact 162,221 frames). No ~1000-episode sim set found; ask Ven/Arjun.
3.0b deliberately stays on the same 100 sim episodes so only the initialization changes.

- [x] 3.1a Headless collection architecture (prompt2assembly-style): launcher mode with NO window and NO stream;
      extract control_gui's scripted pick-and-place (Quickstart sequence) into a GUI-free collector module/ROS node
      that control_gui also calls (no Tk, no virtual display needed for collection).
- [x] 3.1b Clock design (decided 2026-11-13): SIM TIME everywhere for collection (ros2_control, MoveIt, the
      scripted module and the collector `use_sim_time`, on Isaac /clock); frames written on sim ticks via the
      LeRobotDataset API (exactly one frame per 1/30 s sim time, dataset fps = 30 = real), not wall-clock
      lerobot-record. The rate cap becomes a speed knob only: collection uncapped (as fast as the box sustains);
      deployment/eval the same stack capped at 1.0x real time so wall-clock clients (policy, real robot) match.
      Presets `collect` and `deploy`; timing audit proves one frame per tick with state/image/action from one tick.
- [x] 3.1c Diversity and recovery: randomize lego poses (as now) AND cup layout, approach offsets/heights and
      speed; DART-style noise injection (perturb executed motion, record the planner's corrective command as the
      action label) so the policy learns recovery; balanced colours; hold out a fixed eval seed set.
- [ ] 3.1d Pilot 50 episodes, validate (load with LeRobotDataset, feature/fps match with deployment, spot-check
      videos), then full collection ~300 per colour (red/green/blue, ~900) with per-colour task strings, plus a
      data card.
    - [ ] Pilot: 50 accepted episodes, balanced 17/17/16, validated and copied to the Mac (5-episode smoke passed).
    - [ ] Full collection: ~900 episodes; lead schedules after pilot review.
      Searched 2026-11-13: no larger existing sim set on HF (3 accounts + global search), ASU Drive,
      dual-a4500 or rtx6000 (own folders); only teammates' machines unchecked.
- [ ] 3.1f In-process ("battery"-style) collector, if the 3.1d pilot's throughput is too low for ~900+ episodes:
      standalone headless Isaac process, sim stepped only on demand, in-process IK/planning (Lula/RMPflow with cup
      obstacles), cameras via render products/annotators, LeRobotDataset frames per sim tick, several processes per
      GPU (template: prompt2assembly Type-1 headless batteries; pointers in docs/notes/battery-collector-pointers.md). Reuses the 3.1 episode spec. Validated
      against the ROS pilot on the same seeds: success rate, state/action semantics (joint-position targets as
      deployment sends them) and trajectory statistics. ROS 2 + MoveIt stays the deployment/eval/real-robot stack.
- [ ] 3.1e Training throughput before the large run (rtx6000, RTX 6000 Ada 48 GB): the 3.0b run is GPU-bound at
      1.33 steps/s with batch 64, fp32 expert (`use_amp=False`), frozen vision encoder, 16/48 GB used. Benchmark
      bf16 AMP, larger batch with scaled LR, and `torch.compile` on 500-step runs; pick the fastest config that
      matches the fp32 loss curve; size steps by passes over the data (~600k frames for ~900 episodes).
- [ ] 3.2 Fine-tune (more steps/episodes than 10k/100), sim/real mix per the coursework findings.
- [ ] 3.3 Re-run the Phase 1 evaluation matrix.

## Ops / optional (decide individually)

- [ ] NVIDIA driver 570 → ≥580 (fixes cold-launch GPU wedge; reboot — coordinate with prompt2assembly)
- [ ] Box load: prompt2assembly jobs (18–33 load on 28 threads) and its `sudo reboot` at 20:46 on 2026-11-12
- [ ] Stream on/off relaunch comparison (est. RTF 0.31 → 0.37)
- [ ] CPU governor `performance` (≤5–10%)
- [ ] Polished video via recorded-render (re-render a run with path tracing)
- [ ] Check the Mac's offloaded Exploring-VLAs copy (G-DRIVE: unknown commit `5586bab` + 1 uncommitted file)
- [ ] Housekeeping outside the repo: discard the unsent WhatsApp zip draft (us account); commit the
      Career repo's lerobot keep-edit; decide whether `coursework-eee598/` is git-ignored; fix the Mac
      isaacsim state file's stale viewer path (`Isaac Connect.app`).
