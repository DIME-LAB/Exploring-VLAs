# In-process SO-ARM101 collection (ROADMAP 3.1f)

Status: scene export, native GPU/CPU/FK/restore and real two-camera RGB readback
are verified. Eight steady diagnostic updates delivered one camera pair for the
preceding timestamp (within 1 ns), but the first joined boot failed: its first
commanded update delivered no frame, followed by a two-completion burst. The
one-update bound correctly refused it. Texture render-thread preference and
redundant timeline-event changes are offline-tested hypotheses; joined boot,
5+5 episodes, export, throughput and image-equivalence acceptance remain pending.
The runtime and prototype live in `linux-env/scripts/collect_inproc/`; the ROS
collector and `isaac/exts/` remain unchanged.

## Decision and architecture

Use a standalone headless `SimulationApp`, with one main-thread owner of physics,
expert execution, reset and capture. Keep one stage, articulation and two fixed
640x480 render products for the whole run. The proposed live path keeps the
timeline playing and executes one application update per 1/30 s observation,
containing four 1/120 s CPU PhysX steps. At native time t, freeze the measured
state, compute expert/DART, collision-check and apply targets, retaining the real
ACK stamped t. If an immutable frame stamped t is already buffered, validate the
row at t, then advance once. Otherwise advance once and join the delivered frame
stamped t. Both branches return only after one physical advance; only then may
the caller pass the proof to gates or the writer. Native delivery may be zero or
one update late, while camera stamps remain original. No wall pacing or render polling while
the expert runs. Read native clock and unconditional physics callbacks; never
assign time from a row number. Require one completion per product, fresh drawable
frame numbers and consecutive RGB compute counts, with original camera timestamps
matching the retained observation within the shared 1 us tolerance. A missing,
duplicated or skipped frame, epoch crossing or delay beyond one update is a fault.

The canonical clock is the actual Fabric `/ExternalSimulationTime.omni:time`
readback, as used by native graph/camera time. Align the timeline to that clock
without an app update and require their agreement within 10 ns. World.current_time
accumulates float32 PhysX callbacks independently: check its increments with the
shared world_step_errors, as CLOCK_POLICY.absolute_world_native_equality=False
requires. Neither Fabric time nor the camera stamps are rewritten. After reset,
one unrecorded advance establishes the first observation; no pending row is
accepted before its matching pair arrives. Restore writes happen synchronously
between updates and resume a paused timeline without World.play's extra update.
Held-time capture remains a diagnostic mode; its current RGB path failed live.

An extension callback could reuse the live World, but retains its application
frame, GUI, ROS graphs and reset/callback ownership costs. Standalone gives an
explicit stepping entrance and independently owned cleanup. Reuse a flattened
snapshot of the verified sorting scene, including visual assets, cameras, drives,
physics materials, primitive colliders and lights. Export it only inside a granted
window, with a physics baseline, camera paths, source asset digests and code head.
Remove ROS/action graphs and stale SDG writers only from the private loaded stage.
Do not enable the digital-twin extension in the standalone app. This avoids both
hot reload and accidental reuse of a shared articulation. Missing assets, camera
prims, primitives or an incomplete baseline refuse collection.

The first prototype uses the installed 6.1 core APIs already used by our runtime,
with imports after SimulationApp boot. Installed source is authoritative for
signatures; upstream concepts were checked against NVIDIA's
[standalone Python guide](https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html)
and [simulation fundamentals](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/simulation_fundamentals.html).
No shared environment or Isaac installation configuration changes are needed.

## Render alternatives (first-class comparison)

| Mode | Physics pass | Render/write pass | Admission to training |
| --- | --- | --- | --- |
| Live Isaac | Physics + expert + DART; native-stamped camera delivery one update later | Bounded stamp join, RGB encoding and existing exporter | Baseline recommendation; requires 5 normal + 5 re-grasp accepted episodes and same-seed ROS comparison |
| Isaac two-pass | Cameras/render products fully absent; log measured joints, drives, all rigid-body/link world transforms, both camera transforms, metadata and labels at every camera tick | Physics disabled; replay exact poses in a separate Kit, one fixed product per camera, sequential history | Only after live/replay pixel and timing proof; no physical verdict is rerun in pass 2 |
| Browser two-pass | Same pose log | CAD/three.js renderer with calibrated camera, exposure/material compensation | Research option until SO-ARM101 image-gap and throughput are measured; not an interchangeable RTX backend |

Recommend live Isaac for the first accepted dataset, Isaac replay as the next
throughput experiment, and browser rendering as an explicit domain-randomization
experiment. These are provisional choices pending measurements, not predictions
that live rendering wins on total GPU-hours. A faster physics pass alone cannot
justify two-pass collection: report **physics + replay render + encoding/export +
startup/shutdown**, and steady pipeline bottleneck throughput separately.

Aldrin's two-pass proposal is appropriate because no expert, DART, re-grasp or
success decision consumes images. The authoritative physical rows must be kept
immutable. Record each frame's epoch, native sim ns, World time, measured six
positions/velocities, drives, body poses/velocities, every robot rigid-link pose
(including moving jaw and fixed camera mounts), camera world matrices, phase,
clean/executed labels and success evidence. Record a fixed prim inventory and
scene/asset digests. Use compact per-tick pose packets initially, with exact row
hashes; USD animation layers can be derived later. The wrist camera world pose
must be captured rather than reconstructed from a possibly delayed joint topic.
No interpolation, frame deletion or artificial timestamp normalization is allowed.

Replay sets both USD time code (seconds * timeCodesPerSecond) and recorded world
transforms, with consistent xform order and parent-relative conversion. Disable
physics/articulation updates; writing joint targets alone cannot pose links with
physics off. Preserve scale and all static visuals. Verify every replayed link,
object and camera against the recorded matrix before rendering. Do not trust the
Stage Recorder's time origin/segment codes as the native clock. Replay all prior
frames within a segment to preserve RTX history; parallelize by whole episode
first. Frame sharding needs its own temporal warmup/history proof. Motion blur,
TAA and denoiser history can make images differ despite identical instantaneous
poses. Appearance variants retain original physical labels but are tagged variants,
not counted as new independent physical successes.

plan47's read-only evidence: `research/fast-battery-20261018.md`,
`research/cpu-regime-20261020.md`, `research/fast-battery-all-20261020.md` and
`research/cpu-regime-probes-20261020/overlay/square_table_arm_battery.py` under
`~/prompt2assembly/plan-anything`. The warm FR3 battery retained replay digests;
this does not prove SO-ARM101 determinism. Their record-then-render pointers are
`setup/anim-layer-take-howto-20261016.md`, `research/anim-layer-take-history-20261016.md`,
`setup/fmb1-composed-render-20260906.md`, `setup/motionpath-camera-render-20260918.md`,
plus `sim-instrumentation/.../stage_recording`, `stage_take.py` and `replay.rosbag_pose`.
Their 1280x720 render measurement is 0.192 s/frame (0.34 with short-pass startup),
and pixel equality to live RTX is unmeasured. These are priors, not our benchmark.

The browser candidate is the Mac's `~/paper1/site-media/out/isaac-look/`,
`~/paper1/render-recipes/` and `~/.claude/skills/robo-render/`. Its recovered camera
matrices and compensation are task-specific inputs, not SO-ARM101 calibration.
Its CPU result (~420 s / 960 720p frames) does not establish GPU headless-browser
performance. The experiment must render both 640x480 SO-ARM101 views from the same
pose log and compare PSNR/SSIM, RGB errors and side-by-side crops (robot silhouette,
fingers, lego edges, cup interior, glossy table reflections). Reuse CAD and exact
camera optical settings; disclose unsupported RTX effects and material gaps.
If training uses browser images, deployment/evaluation must either use this same
renderer with live measured links/objects and camera calibration, publishing the
existing RGB ROS topics with original sim stamps and a single publisher each,
or explicitly train on mixed/augmented RTX + browser data and validate on RTX
and real observations. Replacing live images with ground-truth-rendered poses at
real deployment would require a perception system supplying those poses; it is
not a free replacement for the real cameras. Do not silently switch deployment.

## Data and correctness contract

Reuse by import: `contract.parameters/corrective_action/timing_audit`,
`dart.SmoothDart/PlannerSchedule/TargetCandidate`, `recovery` geometry/detection/
receipt checks, `tick_contract.TickContract`, `success.initial_arm` and
`settled_containment`, `episode_policy` quality/fault policy, `storage`,
`collection_status`, GPU mapping and `managed_process`. Do not import `collect.py`
just to get its writer/Servo: that imports the ROS process and planner stack.

Rows remain radians internally, exactly as the current raw collector. Feed the
unchanged `collect/export.py` accepted attempt layout: `frames.jsonl`, wrist/top
MP4s, receipt, run config/summary, episodes ledger and affirmative finalization.
Export uses the pinned LeRobot v3 writer, converting measured state and clean
action to degrees in deployment order: shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper_joint. Feature names, 30 fps, HWC uint8 RGB inputs,
task strings and `meta/collection_episodes.json` stay identical. Keep the existing
full state/action/timestamp readback check as an additional validation step.

A held observation is one immutable snapshot. A clean action is
`corrective_action(measured_state, clean_reference)`. Execution is the original
planner reference plus smooth seeded DART, with gripper exactly the clean gripper
command. Do not execute the clipped clean correction instead of the planner path.
Use the same per-operation noise seeds, ramps, derivative bounds, quiet tail and
whole-operation noise scaling. Log both references, offset, filter delta=0,
velocity provenance and physical target readback. Exactly one main-thread apply
belongs to each observation; no ACK/DDS transport is necessary. Keep an epoch and
reset floor, target readback, previous-target persistence and TickContract proofs.
One-process ownership removes queue races, not stale-image or late-read hazards.

Re-grasp retains the shared 2-4 cm seeded miss geometry and parallel clean/miss IK
paths. Record clean true-target corrections throughout the deliberately missed
approach/close/lift, never the injected miss as a label. Confirm the whole close/
lift window using `recovery.detect_miss` (all objects' peak displacement, jaw close
and native tick continuity), reopen, refresh the measured scene, and run the
correct grasp. Populate the exact steps/endpoints/window hashes required by
`recovery.receipt_error`; a recovery tag alone is insufficient. Pilot quota is
5 accepted normal + 5 accepted re-grasp; production retains the shared 18% quota.

Use physical containment from `success.py` and current `cup_geometry.py`, based on
`assets/sorting_collision_geometry.json` (floor top 1.2 mm and twelve tilted box
wall planes). No old radial/floor profile. Check transformed whole-block corners,
settled linear/angular speeds and cup uprightness over 15 consecutive proved ticks.
Require the existing physical lift and home gates. Read fresh geometry/source
hashes after the lead's concurrent cup fixes land.

A collision/planner rejection or physical nonconvergence is a quality rejection
only with a valid checked physical window. Stale data, unsupported geometry,
failed restore, missing camera, corrupted recording or an exception is a terminal
fault. Keep three-consecutive-same-reason stop behavior. Never resume a run with
terminal markers; never mark incomplete rendering/export successful. Two-pass
physics results are provisional until every image exists and re-audits pass.

## Collision safety without per-tick MoveIt RPC

Reuse `grasp_geometry.make_grasp_plan`, `find_reachable_grasp_yaw` and calibrated
`compute_workspace.geometric_ik`. Use direct linear and retract/pan/settle paths
with the same original time law. No new Lula robot description is needed for this
5-DOF chain. A future Lula/OMPL planner is a fallback for rejected paths, not an
excuse to skip validation. Do not change ROS/deployment planning.

The prototype uses in-process oriented-box collision checks on the current native
primitive shapes. Derive joint frames and link-local box transforms from the
loaded USD, install only the SRDF's allowed self pairs and ground-rest allowances,
and verify FK against measured link poses at startup and capture ticks. Refuse
unsupported enabled colliders; never substitute a loose visual bbox silently.
Check self, ground, all cup boxes, all other legos and carried-block volume. Allow
only designated finger-pad contact with the selected lego during grasp, with an
explicit phase policy. A carried block is a planning envelope relative to the
measured gripper, not a simulated fixed joint or teleported object.

Use shared `command_geometry.checked_segment/final_command` to validate both
measured-state→executed and prior-drive→executed sweeps at <=0.01 rad. Preflight
the entire arm operation and DART scale before stepping; at each tick refresh
moving-body obstacles and recheck actual tracking sweeps. This retains target and
sweep checks while removing service serialization and MoveIt scene round trips.
Discrete joint samples are not a mathematical continuous-collision proof: retain
the current spacing, report this limit, and conservatively inflate boxes by a
link-motion bound between samples before claiming stronger coverage. An unsupported
shape or unverified joint map is a fault. Collision refusal never clips a command
or silently eliminates DART on individual ticks.

## Reset, process, GPU and storage ownership

Restore existing bodies in place, all orientations, six arm positions, all arm
and rigid velocities, drive targets and scene bookkeeping; no new_stage or rigid
body insertion beside a live articulation. Increment epoch without rewinding the
native clock. Verify immediate float32 readback and a fresh six-tick stable-home
window; establish the drive/clock baseline before the first recorded tick. Seed
Python/numpy and deterministic layout draws independently; record code/input hashes,
physics knobs, solver settings and seeds. Numerical reproducibility is measured
by same-seed independent run digests, not promised across platforms.

Launch with a physical GPU UUID in CUDA_VISIBLE_DEVICES and an independently
resolved Vulkan renderer activeGpu; CUDA visibility does not select Vulkan.
Disable multiGpu, GPU PhysX and wall rate limiter; verify CPU dynamics and renderer
readback after World construction. Keep persistent per-GPU shader/OptiX/CUDA caches,
never clear shared caches or replace HOME. Separate instance tmp/log/run roots.
No CPU pinning/cgroup changes without a measured benefit; cap worker threads first.
Use the shared storage floor/output confinement; owned-process records bind PID,
start ticks and PGID. Encoder children are owned explicitly. SimulationApp exit 0
is insufficient: require final receipt, export validator and end marker.

## Phases and acceptance experiment

1. Design and CPU tests: clean/executed separation, exact TickContract and reset
   baseline, skipped ticks/readback refusal, seeded DART, replay digest/inventory/
   epoch continuity. Run shared success/recovery/safety/timing tests as well.
   Report design, paths and test proof to `so-arm-101-vla` before live work.
2. Implement isolated prototype and launch/preflight commands without starting
   Isaac. Add scene snapshot, direct physical stepping/capture, state replay,
   collision checks, expert stages and compatible raw writer/receipts. CPU tests
   are logic verification only; docs must say which adapters remain live-unproved.
3. Ask lead for an exclusive sim window, after w_scene and ROS smoke. Only then
   export the baseline and run startup probes, single-pass and physics-only/replay
   comparisons. Lead must grant the window explicitly; elapsed time is not approval.
4. Obtain 5 accepted episodes per kind (balanced colors where possible), bounded
   attempts and physical failures retained. Run the exact exporter, full raw-row
   timing/recovery audits and LeRobot readback. Compare with ROS on identical seed,
   starting snapshot and commands where possible; different planner paths are
   trajectory-distribution comparisons, not pixel-fidelity tests. Save synchronized
   two-camera contact sheets and videos, same-pose RGB PSNR/SSIM, peak/mean joint
   tracking, action delta/velocity/acceleration, lift, slip and placement yield.
5. Benchmark startup, reset/planning, four physics steps, renderer submit/wait,
   GPU→CPU copy, expert/DART, collision, contracts, encoder queue/write/drain and
   export separately (median/p95/total). Report stepped and end-to-end RTF including
   rejected attempts, accepted episodes/hour and GPU memory/load. Target >1 RTF
   per instance is an acceptance target, not an offline estimate. The 35-42 ms
   Isaac-frame prior already exceeds a 33.33 ms frame budget; simply deleting
   orchestration cannot prove >1. Incorporate w_scene results when available.
6. Only after N=1 correctness, compare N=2 aggregate accepted throughput on the
   assigned GPU(s). Two-pass/browser results need separate live/replay image proof.
   Publish artifacts to the Mac's
   `coursework-eee598/roadmap-2026-11/phase3/3.1f-inprocess-collector/` with README,
   tests, per-stage JSON, comparison frames and validator reports. No HF upload.

## Prototype status and review

The isolated prototype includes scene export, CPU stepping, native state/drive
readback, synchronized RGB capture, scripted normal/re-grasp stages, primitive
collision sweeps, raw MP4/JSONL recording, full numeric LeRobot readback, an owned
launcher/watchdog and experimental pose replay/comparison commands. It consumes
ROS `initial_physical` receipts to match seeded layouts; autonomous layout generation
is not implemented. Direct paths are implemented; retract/pan/settle and OMPL
fallback remain proposed if measured yield requires them. No live success or speed
result is implied by these entrances.

The source review identified shutdown/finalization and mm→m collider scale risks.
Box extraction retains inherited scale in a rigid metre body frame. Kit uses
fast shutdown after draining encoders and closing the backend; the external owner
finalizes only after the producer exits successfully and identity-bound descendant
cleanup has no remaining processes. Producer PID, invocation, start time and
summary digest bind the worker's readiness record. A zero exit without these
records is refused. Graceful Kit teardown stalled during the boot window, so the
owner performs finalization outside Kit. Regression checks cover millimetre bodies,
real FFmpeg encode/decode, failure readiness without a summary and lifecycle
certificates. The review also prompted exact ROS DART seed/phase rules,
measured pause/encoder-retirement fields, recovery audit for physics candidates,
and camera poses derived from measured parent links.

`playing.py` follows the verified camera setup in `~/w_31f/bench.py`: retain the
existing active viewport with updates disabled (create an invisible one only if
absent), apply OmniSensorAPI tickRate 30, CPU RGB annotators as in `record.py`, RenderingManager dt
1/30 and timeline target 30, play and warm up. Installed SimulationContext.step
with render=True delegates to one app.update at these nonzero dt values. The
adapter refuses extra/missing physics steps or renders rather than polling a stale
buffer. ReferenceTime is global, so it is supplemented by per-product drawable
frame identity, RGB compute count and simulation NEW_FRAME completion count. Read
frame metadata in the drawable callback using its actual result_handle; handle 0
is invalid here and the optional SWH field is absent. Copies
retain the producer's rational timestamp, including permitted sub-microsecond
rounding differences. Whether these native fields reliably identify both pixels
after every update is still a live-verification requirement. Experimental replay
and held writer-gate capture remain non-exportable until verified.

The offline suite has 45 tests, including immediate/delayed control-image order,
the measured eight-frame latency trace,
2,000 float32 World/native-clock frames, earlier-ACK/frozen-state preservation and
late/missing/duplicate/epoch-crossing join refusal. The probe uses its own
BOOT_PROBE_FINISHED marker with episodes=0; collection export is not admitted.

The wall watchdog accepts 900 s startup, 120 s without collection progress and
180 s graceful shutdown, with pidfd cleanup and an affirmative producer-bound
finalization check. Heartbeat touching adds no fsync per frame. No existing Isaac
or prompt2assembly process is adopted or stopped.

## First window result and next gate

The 2026-11-14 boot-only window ended at 07:23:13Z. The held scene export contains
11 robot links, nine rigid scene bodies, 65 enabled boxes and ten joints. Camera
paths came from the current collector configuration and are stored in the export
manifest. All eight standalone attempts passed GPU UUID/Vulkan/NVML assignment,
nice/domain, CPU dynamics, native joint FK and immediate baseline restore checks.
Four-step stepping executed before RGB refusal, but those probe versions did not
persist its numerical evidence before failure; there is no complete cadence pass.
No paired RGB frames were accepted. The first six attempts observed the wrong
NEW_FRAME event channel; their zero event count proves nothing about rendering.
Attempts seven/eight used the simulation channel and timed out with zero observed
completions. No renders-per-advance ratio or RTF can be claimed.

Paused tick-rate sensors and an extra viewport/Hydra engine are plausible causes,
supported by w_scene's playing-timeline reference and plan47's camera-shapes
notes; they were not experimentally isolated in that window. Its same-update
proposal is superseded by the bounded stamp join below. Use
`launch.py preflight|start --probe ... --step-mode playing` with a fresh
output/runtime and require the joined boot gate. If it fails, retain the fault
and release; do not begin episodes. Five normal/five
re-grasp episodes additionally require the fresh ROS-smoke fixtures and the lead's
explicit episode window. Cleanup proof is
`/mnt/a4500-backup/exploring-vlas/inproc/window-20261114-cleanup.json`: nine owned
groups stopped, remaining=[], ports 8781/8782 closed, no owned NVML contexts,
GPU0 at 21 MiB. The companion Mac report contains raw probe receipts and logs.

## Playing-window result and capture settings

The second boot-only window ended at 08:00:26Z on 2026-11-14, with all four owned
groups stopped, ports closed and no owned NVML contexts. The v2/v3/v4 probes
observed respectively 109/121/120 render completions per camera. No RGB pair
passed: both GPU and CPU read paths returned empty uint8 buffers, with RGB node
compute count zero. The v4 warmup diagnostic was empty before its final restore,
so this cannot be explained solely by the post-reset camera state. v3/v4 retained
one actual four-PhysX-callback/four-World-step advance; World time was
4.066666878759861 s and timeline 4.066666877 s. This is partial physical evidence,
not a complete cadence/freshness acceptance or renders-per-tick result.

Cold v2 startup took 343 s, including about five minutes of RTX compilation.
The installed GPU-foundation source defaults `driverShaderCachePath` to the private
portable directory; the launcher now pins it to a persistent per-GPU directory.
Owned stopped-v2 cache contents were retained there. Later v3/v4 runs ended in
22/25 s; these are startup/fault wall measurements, not collector throughput or a
controlled cache ablation. Native initialization receives its own 900 s heartbeat,
followed by the 120 s progress timeout. Product updates and timeline auto-update
are disabled across restore's pause/play updates, preventing unrecorded reset
renders. This guard did not repair the RGB computation failure.

The lead's capture-on-play hypothesis has direct source support: installed
`Orchestrator.reset()` copies `/persistent/omni/replicator/captureOnPlay` into
`/omni/replicator/captureOnPlay`. The prototype previously set only the runtime
flag; the failed probes did not record either readback, so overwrite is not yet
an established cause. Set both flags true at boot and before camera attach/play,
verify both before every recorded step, and retain warmup readbacks. Private Kit
settings remain isolated; no default user configuration is edited. Tests cover
the persistent-to-runtime reset behavior and refusal when a flag turns off.

Set RenderingManager dt before products create Fabric histories. Following the
working `~/w_31f/replay.py`, ensure `/ExternalSimulationTime` exists and record its
`omni:time` value. During live physics the simulation manager owns that value:
read it as the canonical graph clock and check World increments independently.
Kinematic replay may explicitly stamp it only with physics off.

## RGB readback and measured delivery phase

The third window ended at 08:24:10Z on 2026-11-14. All seven owned groups stopped,
ports closed, no owned NVML contexts remained and GPU0 returned to 21 MiB; proof
is `inproc/playing-window3-20261114-cleanup.json` under the canonical storage root.
Both capture flags read true in v5, yet RGB compute remained zero. Capture-on-play
overwrite alone therefore does not explain the failure. Match the known-good
`w_31f/run_arm.py` legacy pipeline (`srtx/enabled=false`, async rendering false).

v7's SDG diagnostic found DispatchSync enabled with empty valid-time lists: its
reference-time inputs advanced while RGB execution stayed blocked. The documented
`disableAnnotatorGate=true` compatibility setting bypasses that internal filter;
our own bounded stamp/frame/copy/epoch checks retain the acceptance guarantee.
v8 onward produced real 480x640 RGBA buffers from both cameras. Diagnostic warmup
PNGs show the exported scene; they are not accepted episode frames or a ROS match.

v11's fault-only trace demonstrated consistent one-update delivery over all eight
updates: camera RefTime matched the pre-update observation within 0–1 ns; each
update advanced four physical callbacks/World steps and completed one render per
camera. RGB compute counts were 121–128, and both views had identical drawable
frame numbers (275,293,295,...305). No camera timestamp was changed. The trace
and source digest are retained in `collect_inproc/fixtures/playing_v11_trace.json`.
The previous same-update probe correctly rejected these deliveries; zero dataset
rows or episodes were accepted during the window.

`Capture.prepare` retains the measured state, clean/executed labels, geometry
evidence and actual target ACK at t. `Capture.complete` joins original camera
stamps to that retained state within one update and invokes the same shared row
validator. It cannot issue the command twice or finalize a pending/reset-crossing
ticket. Expert/home/recovery calls still return the pre-update observation after
advancing once, preserving control ordering. The initial advance after restore
primes delivery; the first row is still exactly one tick after the restore floor.
These changes are offline-tested against the measured trace. Next window must
verify joined proofs, a collision-checked pan movement, a paused second reset and two further joined frames,
no work while held between updates, and successful producer exit/finalization.
Five normal/five re-grasp episodes, export/readback, ROS comparison and RTF remain
pending fresh fixtures and an explicit episode grant.

The `--motion-check` boot option moves pan by 0.12 rad through shared clean
corrections and the ordinary collision/ACK entrance. It retains actual drawable
view matrices and measured camera world transforms at observation and arrival.
Require measured pan motion >0.01 rad, each wrist view within 2e-4 (maximum matrix
entry error) of the observation pose, and at least one moving frame clearly
closer to observation than arrival (>1e-4 difference). Boundary PNGs also require
a visual check; stamps alone do not establish physical pixel phase. These pose
thresholds are diagnostic gates to verify live, not a measured fidelity result.

The lead's current read-only fixture root is
`/mnt/a4500-backup/exploring-vlas/datasets/smoke-primitive-native-tick-v3/`.
Accepted normal fixtures are attempt-0000 (330000, red) and attempt-0002 (330002,
green_2x3_0); re-grasp fixtures are pending. The earlier designated fixture root
was not created. A normal pilot can use these two seeds after boot acceptance;
it does not fulfill 5+5. Retries exclude already accepted seeds, and summaries
record accepted seed sets so a repeated success cannot fill a distinct-seed quota.

## First joined boot result

`c4f6396` was exercised in an explicitly granted boot-first/normal-pilot window.
The worker failed after 26.95 s at its first commanded join: observation
4.066666667 s advanced physically to 4.1 s with four substeps, but both cameras
still held RefTime 4.0 s and copy count120, with zero new completions. The next
fault-only diagnostic update produced two completions per camera and copy count122
(latest RefTime4.099999999 s), followed by seven regular one-completion updates.
Only the latest buffer of the burst was read, so no unique timestamp sequence is
claimed for the two individual deliveries. No pending row was finalized, no
normal pilot started, and the motion/reset/success-exit gates were not reached.

Sim released early at 09:13:08Z on 2026-11-14. Proof is
`inproc/joined-window-20261114-cleanup.json`: one owned group stopped, remaining=[],
ports8781/8782 closed, no owned NVML contexts/errors, GPU0 at21 MiB. The earlier
steady trace does not establish a uniform startup bound.

The next probe explicitly requests `hydra_texture.set_async(False)` before
annotator attach, retaining `get_async()` after warmup. Installed API documents
this as a render-thread preference; readback alone cannot prove synchronous
delivery. App/Replicator async settings did not establish the texture preference
in the failed run. Restore now emits play/commit only when paused, and timeline
alignment avoids no-op current-time writes while the native clocks already agree.
These changes retain the same strict join and are not claimed to fix the burst.
A new granted probe must establish actual delivery, motion pixel phase, paused
reset and successful exit before episodes. If variable-latency bursts remain,
polling only the latest annotator buffer cannot admit them; per-frame immutable
capture and a broader bounded control/image queue would need a separate design
and contract review. No delay is hidden by dropping or restamping a frame.

The bounded join also retains a frame already delivered at its own native time,
because a synchronous render preference may change delivery phase. Such a frame
is consumed after its actual command ACK at the same native tick; the tick still
advances physics exactly once before returning. The measured delayed phase
continues to join after that one advance. Tests exercise both orders and reject
duplicate or skipped observation stamps. Neither timing mode is assumed to work
live from the preference readback alone. The next sim window is queued after the
ROS collector's 50-episode pilot; no new launch is authorized by this document.

The lead has supplied a newer read-only fixture source with five accepted normal
episodes: `datasets/smoke-primitive-reset-v4/` under canonical storage. Use its
completed receipts and the pilot output when the lead grants the episode window;
re-grasp input provenance remains to be supplied. Older v3 seed fixtures remain
comparison evidence, not a requirement to use superseded reset behavior.
