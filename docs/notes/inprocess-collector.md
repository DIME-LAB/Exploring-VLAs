# In-process SO-ARM101 collection (ROADMAP 3.1f)

Status: the production fixed-base correction is verified, with unchanged six-joint
order and zero measured base drift in the real ROS pan/lift sweep. A fresh held
scene is exported. A 32-tick motion/reset diagnostic proves the SDG post-render
camera matrix matches the retained native observation (maximum error 1.71e-7) and
its compute counter stays aligned with RGB. SDK drawable matrices are one update
older and are retained only as diagnostics. The corrected admission path requires
fresh post-render metadata and checks camera pose on every row before finalizing
its tick proof. Its final native boot/success-exit check is queued after the ROS
smoke. No 5+5 episodes, export/readback or collector RTF are accepted yet.
Runtime/prototype files live in `linux-env/scripts/collect_inproc/`. Shared ROS
collector and extension code are unchanged; the production robot asset's two
root API opinions are corrected.

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
frame numbers, consecutive RGB and post-render CameraParams compute counts, with original camera timestamps
matching the retained observation within the shared 1 us tolerance. A missing,
duplicated or skipped frame, epoch crossing or delay beyond one update is a fault.

The canonical clock is the actual Fabric `/ExternalSimulationTime.omni:time`
readback, as used by native graph/camera time. Align the timeline to that clock
without an app update and require their agreement within 10 ns. World.current_time
accumulates float32 PhysX callbacks independently: check its increments with the
shared world_step_errors, as CLOCK_POLICY.absolute_world_native_equality=False
requires. Neither Fabric time nor the camera stamps are rewritten. After reset,
bounded unrecorded priming first establishes steady camera cadence, then
one boundary advance establishes the first observation; no pending row is
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

Reuse `grasp_geometry.make_reachable_grasp_plan` and calibrated
`compute_workspace.geometric_ik`. The shared helper couples narrow-face yaw
(only equivalent0/+pi/-pi choices), TCP centering, safe approach offset and
projected-width jaw opening/closing. Use its returned solutions and jaw values
together; do not combine raw `make_grasp_plan` with a separate yaw fallback.
The prototype implements direct linear paths with the original time law.
Retract/pan/settle paths remain a proposed fallback for rejected direct paths. No new Lula robot description is needed for this
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


## Implemented pipeline and verified limits

The prototype consumes exact ROS `initial_physical` snapshots and params to match
seeded layouts. Autonomous layout generation is not implemented. `expert.py` uses
the shared reachable grasp result as one coupled specification; recovery uses the
shared true/miss geometry and its selected jaw values. Receipts retain effective
grasp specifications and DART programs. Direct paths may be rejected; live normal
and recovery yield remains to be measured.

`playing.py` retains one paused viewport engine, two 640x480products, legacy CPU
RGB/ReferenceTime/CameraParams annotators,30 Hz sensors and four 120 Hz physics
substeps per app update. Both captureOnPlay settings are true. Native PhysX Fabric
is enabled before reset, USD pose/velocity writeback is disabled, and physical
transforms are explicitly published before the single app update. Publication
must not step physics, change either clock or render. Every snapshot requires
valid articulation/body/link physics handles; stale USD fallback is a fault.
Record all settings and per-stage timings. Held-time capture and replay remain
experimental and cannot be exported as accepted data.

After every immediate verified restore, hold the restored drive targets and
discard at most 30 unrecorded updates. Require three consecutive previous-tick
camera pairs with fresh RGB/CameraParams counts, correct original timestamps and
four physical substeps. Physics/drive faults are terminal; exhausting camera
stabilization is terminal. Keep the raw restore proof and every discarded update.
Validate the immediate restore against the preceding TickContract before opening
a fresh recording ledger at the measured end of priming. The ordinary first
observation remains one tick after that boundary. Even its discarded boundary
image must pass the strict producer ledger.

The authoritative render camera matrix comes from legacy
`omni.replicator.core.CameraParams`, which depends on
PostProcessRenderProductCamera/SdRenderProductCamera and actual renderResults.
Require its compute count to advance exactly 1 alongside RGB for each camera.
`Playing.render(expected=T)` compares this matrix with the frozen physical link
poses and static camera bindings, with maximum matrix-entry error<2e-4 for both
views. A mismatch faults before `Capture.complete`, so no invalid row reaches a
success gate or writer. SDK drawable IDs still supply progress diagnostics; SDK
view matrices are not authoritative for the RGB camera.

The 32-tick native diagnostic at9857b38 measured SDG camera-pose error<= 1.71e-7
through pan motion and a paused second reset. CameraParams compute counts equaled
RGB copy counts on every captured tick and each advanced1 within an epoch. SDK
view matrices were one tick older, with errors up to0.030706 during motion. The
actual packets and source receipt SHA are preserved in
`fixtures/camera_params_32_trace.json`. Regression checks admit that trace and
refuse substituting its older SDK view. No image or timestamp was changed.
For probe reports, `camera_pose_phase` now means the authoritative SDG comparison;
`drawable_pose_phase` retains the SDK comparison. Historical reports before this
change used `camera_pose_phase` for SDK metadata, and are not directly comparable
by field name. The final native boot must still pass the corrected gate and prove
successful owner-side finalization.

The production asset now authors its root API on the world fixed joint. Native
production verification confirms fixed_base=true, root DOFs0, joint DOFs6, links 11
and the exact six-name order. Six real ROS subscriber/ArticulationController
pan/lift goals measured maximum root drift0m/0rad across 272 physical callbacks.
Shared collection_runtime installation also resolved the same joints. The fresh
held export is `inproc/scene-fixed-root-20261114/`, including the native audit and
camera paths read from the collector's actual configuration. The asset-only
correction preserves every other authored opinion and both relative external
references; extension and shared collector code are unchanged. See
`inprocess-collector-base-root.md` for source digests and independent reviews.

The owned watchdog accepts900 s startup,120 s without capture progress and180 s
shutdown, with the lead's granted wall deadline taking precedence. After encoder
and backend drain, the child publishes invocation/PID/summary-digest readiness and
fast-exits. Only the external owner can finalize after successful child exit and
identity-bound descendant retirement. Zero exit alone cannot produce acceptance.
The boot marker is BOOT_PROBE_FINISHED, kind=boot_probe, episodes=0; it is never a
dataset. Tests cover real FFmpeg encode/decode, lifecycle certificates, physical
handle faults, frozen state/real ACK preservation, producer freshness, current
camera pose, original clocks, collision geometry, seeds and writer layout.

Partial diagnostic timing from16 boot ticks (means,ms) was app update33.62,
camera read4.91, combined physics/render wrapper38.79, state0.91, clean correction
0.32, collision8.78, target write0.28 and join/hash/contract9.65. App/camera values
are nested in the wrapper and must not be added again. This excludes startup,
remaining host work and the dataset writer: no end-to-end collector RTF or>1x
result is claimed. Full profiles, accepted throughput and all three render modes
still need measurement.

## Remaining acceptance work

GPU 0is currently owned by the fixed-base ROS smoke. The final corrected boot and
episode window require a fresh lead grant after it finishes. Read-only fresh
fixtures appear under `datasets/smoke-descent-clearance-fixed-v1/`; accepted normal
attempts 0000/0001 use seeds 330000/330001. Check completed receipts and provenance
when selecting fixtures; do not modify them. Earlier floating-base fixtures do
not fulfill the corrected-physics acceptance experiment. Re-grasp fixtures remain
pending. No native successful finalization,5+5accepted dataset, unchanged
export/readback, same-seed ROS frame comparison, renderer-gap metrics or collector
RTF has yet been accepted. The lead retains a live GUI check before merging main.
