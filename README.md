# Exploring-VLAs

A digital twin of the **SO-ARM101** 5-DOF arm in **NVIDIA Isaac Sim 6.1**, driven
by a fine-tuned **SmolVLA** vision-language-action policy. One script brings up
the whole thing on a Linux GPU box:

| Piece | What it is | Runs on |
|---|---|---|
| Isaac Sim + `soarm101-dt` extension | Physics, cameras, legos and cups scene, ROS 2 publishers (`isaac/exts/soarm101-dt`) | the box, GPU 0 |
| omni-kit-mcp bridge | A small TCP server inside Isaac that serves the extension's scene commands as `soarm101.<name>` tools (port 8767), plus a control panel in the Isaac window (`omni-kit-mcp/` submodule) | inside Isaac |
| ROS 2 control stack | MoveIt + ros2_control, plus the opt-in `control_gui` (Tk window) and RViz (`vla_SO-ARM101/`) | the box |
| SmolVLA policy | lerobot async policy server + robot client: async action chunking with real RTC prefix guidance (`lerobot/` submodule) | the box, GPU 1 |

A Mac (or any other machine) only *watches* the sim over Isaac's WebRTC
livestream and, optionally, scripts the bridge through an ssh tunnel.

> Mac-native Gazebo backend, dataset recording and the planner internals are
> covered in [`CLAUDE.md`](CLAUDE.md) and the docs it links.

Standalone LeRobot ROS 2 plugins and the opt-in RunTask node live in
[`packages/`](packages/README.md). Current launchers retain the pinned fork until
ROADMAP 2.7; package installation, native checks and the optional Isaac panel are
documented there.

## Prerequisites (the Linux box)

- Ubuntu 22.04 with an NVIDIA GPU (tested: 2x RTX A4500). One GPU also works:
  add `SOARM_POLICY_GPU=0`.
- NVIDIA driver ≥ 580 recommended (Isaac Sim 6.1 recommends 580.95). On 570.x,
  first (cold-cache) launches can wedge the GPU right after the wrist-camera
  publisher starts or at `Playing scene` (~1 in 4 in our tests). The launcher
  detects this and stops with exit 3 and a clear message (no automatic retry);
  rerun `up`. Warm-cache launches were 8/8 reliable.
- **Isaac Sim 6.1** pip-installed into a venv at `~/env_isaacsim61`
  ([NVIDIA pip install guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)).
  Elsewhere: `export ISAACSIM_BIN=/path/to/venv/bin/isaacsim`.
- ROS 2 Humble at `/opt/ros/humble`, [`pixi`](https://pixi.sh) (a conda-style
  env manager, used for lerobot's Python 3.12 env), git, ~50 GB free.
- Only for the optional windows (`control_gui`, RViz; on by default without
  `--stream`): a logged-in desktop session on the box. They open on the box's
  own screen, never in the stream. Headless runs (`--stream`) don't need one.
- Blender is not needed: the baked ArUco marker meshes are committed under
  `isaac/exts/soarm101-dt/assets/aruco_markers/meshes/`. Blender 4.x is only
  needed to regenerate them (delete a file, set
  `SOARM_BLENDER=/path/to/blender-4.x`, run `soarm101.quick_start`).
- For streaming: Tailscale on the box (its tailnet address is advertised to
  the viewer), or set `ISAACSIM_STREAM_IP=<an address the viewer can reach>`.

## On the box itself

```bash
git clone --recursive https://github.com/DIME-LAB/Exploring-VLAs.git
cd Exploring-VLAs
bash linux-env/scripts/bootstrap.sh          # submodules, pixi env, colcon build (once)

bash linux-env/scripts/deploy_sim.sh up      # Isaac window + scene + control stack + policy
bash linux-env/scripts/deploy_sim.sh status
bash linux-env/scripts/deploy_sim.sh down
```

`up` runs four steps in order, each skipped if its piece is already running,
and prints every log path: Isaac (waits until the bridge answers), the scene
(`soarm101.quick_start`), the control stack, then the policy. The first `up`
downloads the model from Hugging Face, so give it a few minutes.

**The first Isaac launch on a machine is slow.** It compiles RTX shaders,
which takes ~2-7 min (the Isaac log shows `Waiting for RtPso async group async
compilation: N seconds`). It usually lands during the first scene build
(step 2), while Kit answers nothing. It's a one-time cost per GPU/driver/Isaac
version and later launches skip it, so don't kill it; `up` waits it out (up to
15 min, `SOARM_QUICK_START_TIMEOUT`) and `isaacsim_launch.sh status` shows the
progress. If another Isaac/Kit process is running on the same box, you may see
`Disabling key-value database because another kit process is locking it` and
lose cache reuse. The caches are per-user and safe to keep:
`~/.cache/ov/shaders/` and `~/.cache/ov/DerivedDataCache/`.
More detail: [NVIDIA forum thread](https://forums.developer.nvidia.com/t/388276).

## From a Mac

Isaac Sim runs on the Linux GPU box; the Mac is the screen.

1. Install the **Isaac Sim WebRTC Streaming Client** for macOS (from the
   [Isaac Sim livestream clients page](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)).
2. Start everything headless with the stream on:

   ```bash
   ssh <box> 'cd ~/path/to/Exploring-VLAs && bash linux-env/scripts/deploy_sim.sh up --stream'
   ```

   It ends with `Stream: connect the WebRTC Streaming Client to <address>`.
   Everything is detached, so the ssh session can close. This is headless:
   the control stack runs without `control_gui`/RViz (they could only open on
   the box's own screen). Add `--gui` / `--rviz` if someone is logged in at
   the box and wants them there.
3. In the Streaming Client, enter that address (the box's tailnet IP or
   MagicDNS name) and connect. You get the whole Isaac window: viewport,
   the SO-ARM101 extension window, and the omni-kit-mcp panel, all clickable.
4. Optional, script the bridge from the Mac. The client is one stdlib-only
   file, so copy it and tunnel the port:

   ```bash
   scp <box>:~/path/to/Exploring-VLAs/omni-kit-mcp/kit_mcp/client.py .
   ssh -N -L 18767:127.0.0.1:8767 <box> &
   python3 client.py list_tools --host 127.0.0.1 --port 18767
   python3 client.py soarm101.randomize_object_poses '{"required_visible_color": "blue"}' --host 127.0.0.1 --port 18767
   ```

   Exit codes: 0 ok, 1 the tool reported an error, 2 bridge unreachable.
5. Stop: `ssh <box> 'bash ~/path/to/Exploring-VLAs/linux-env/scripts/deploy_sim.sh down'`.

## Picking a model

```bash
deploy_sim.sh up --model mix-75s25r
deploy_sim.sh up --model your-hf-user/your-smolvla      # any HF repo id or local pretrained_model dir
deploy_sim.sh up --task "Pick a red lego and place it in red cup"
```

| Preset | Hugging Face model | Fine-tune data |
|---|---|---|
| `sim` (default) | `anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200` | 100 sim episodes |
| `mix-75s25r` | `anirudhrani/smolvla_sim_100ep_75S_25R_fft__10ksteps_h200` | 75% sim / 25% real |
| `mix-25s75r` | `anirudhrani/smolvla_sim_100ep_25S_75R_fft__10ksteps_h200` | 25% sim / 75% real |
| `mix-50s50r` | `anirudhrani/smolvla_blue_sort_50sim_50real_10k` | 50 sim + 50 real |

`deploy_sim.sh models` lists the complete preset menu. It and the GUI read
`MODEL_PRESETS` in
[`model_presets.py`](vla_SO-ARM101/src/so_arm101_control/so_arm101_control/model_presets.py).
Add a local checkpoint with one row, for example
`"base-ft": ("~/w_ckpt/<run-name>/pretrained_model", "", "{}"),`;
the path must contain the model’s `config.json`. Restart the GUI after adding
presets. Inference defaults to 17.4 client FPS, rgb8 cameras and real RTC;
its Method radio selects Custom or Async, and the RTC checkbox controls Async.
Both methods can record a dataset; Stop saves the live episode before recovery.

To switch models without restarting Isaac: `deploy_sim.sh down --policy`, then
`deploy_sim.sh up --model <other>`. The same presets are in `control_gui`'s
Inference tab; if you'd rather start inference from there, bring the stack up
with `up --gui --no-policy` so two policies don't drive the arm at once.

## Policy timing and episodes

The policy server and robot client default to **17.4 Hz**, selected from the
recording audit, with a FollowJointTrajectory duration of `1/fps` (57.5 ms).
`deploy_sim.sh up --fps 19.4 --action-duration-s .1` overrides both. Camera
capture, RGB decoding and observation RPCs run in a persistent child process;
an independent action thread dispatches against absolute deadlines. Deployment
defaults to `--observation-process --dispatch-priority 10`: only that native
thread uses FIFO priority 10 (requires scheduler permission or the enabled
`sudo -n chrt` fallback). `--dispatch-priority 0` selects best-effort scheduling;
`--no-dispatch-sudo` forbids sudo. The stable Python heap is frozen before active
episodes to keep full GC pauses out of the action clock. Dataset metadata remains
30 fps; async actions are indexed steps and are not resampled from that metadata.
A 45-second live check reached **17.3999 Hz**, with zero missed/repeated actions
and no empty slots after startup (load 16.39 on 28 logical CPUs). See the
[episode harness README](linux-env/scripts/eval/README.md) for timing evidence.

RTC is enabled by default. The server supplies the previous normalized,
unexecuted chunk tail and measured inference delay to the policy; `--no-rtc`
selects the unguided comparison. A recorded-episode replay at 17.4 Hz reduced
mean raw boundary jump 9.06→4.77° L2, with nonzero guidance correction. See
[the RTC/timing report](linux-env/scripts/eval/rtc-report.md) for the method,
plot data and limits.

With Isaac and controllers already running, use the
[episode harness](linux-env/scripts/eval/README.md) to home, randomize blue
visible, run the persistent policy client, score cup placement and save
viewport MP4s plus CSV/JSONL. The server loads the model once for the run.

## Windows: `--gui` / `--rviz`

`control_gui` (the Tk control panel) and RViz are opt-in. They are on by
default for a local-window run and off with `--stream`; `--gui`/`--no-gui` and
`--rviz`/`--no-rviz` override either way. Without them the control stack is
ros2_control + MoveIt + the pose publishers, and `up` waits for the
`arm_controller`, `gripper_controller` and `joint_state_broadcaster` to be
active; the policy only needs those. Things that do need `control_gui`: its
Record Sim and Inference tabs, `vla_SO-ARM101/scripts/sim_reset.sh`, and the
`/so_arm101_control_gui/*` services. When a window is asked for, the box's X
display must be usable; if it isn't, `up` stops with that message rather than
starting the window somewhere you can't see.

Saving any file under `isaac/exts/` hot-reloads the extension in a running
Isaac Sim and stops its publishers. Edit there with Isaac closed, then relaunch
with `isaacsim_launch.sh close` and `deploy_sim.sh up` (add your stream/GUI flags).

## Stopping

`deploy_sim.sh down` stops the policy, then the control stack, then Isaac,
and only the pieces that `up` started (an Isaac or control stack you started
yourself is left alone). `down --policy` stops just the policy.

## Simulation cameras

Combined CPU/camera fixes measured 31.08 ms/frame and RTF 0.5363 unthrottled
(65.07 s, load 20.81), versus 57.14 ms / RTF 0.2917 before (load 17.52).
These are shared-box observations, not isolated benchmark claims.

Both image topics publish **640×480 rgb8**, one image per 60 Hz simulation
frame, paced by the selected wall rate. Headers use simulation time and reset
with `/clock` on timeline stop. The wrist uses its render product and
`ROS2CameraHelper`; the workspace helper attaches to the **existing stream
viewport render product**, so the topic and stream share the same view and
no second workspace render is needed. There is one publisher per topic;
manual viewport publishers reject those reserved topics.

LeRobot's real Jazzy `ROS2Camera` decoded 20 live frames per camera and matched
equivalent RGBA decoding byte for byte. The recorder and inference camera
defaults use rgb8. In 65 s paced and unthrottled checks, all received image
stamps were unique with exactly 1/60 s increments and no frame gaps.
Against the old viewport paths, 32-frame temporal means measured wrist
PSNR 47.61 dB / MAE 0.78 and workspace 73.17 dB / MAE 0.029 (8-bit units).
Decoded `rec_222812` AV1 samples (32 wrist, 48 workspace frames) retained the
same camera framing and color hues: first-home-frame median blue/green/red
hues differed by at most one OpenCV hue unit. Whole-image RGB statistics vary
with object layout, robot motion and compression; they are not pixel equality.
No camera pose, intrinsic, lighting or material values were changed.

When removing a camera helper, also remove its writer under
`/Render/PostProcess/SDGPipeline/`; deleting only the graph can leave a live
publisher in Isaac 6.1. Inspect publisher counts through ROS, since the
viewport-publisher list only describes the optional manual capture path.

## Known issues

- **Cold rendering warm-up.** Robot geometry is de-instanced at scene build,
  including visual/collision reference roots that can share Hydra prototypes.
  Three independent empty portable roots rendered the robot: 36 meshes,
  zero instance proxies and zero `cannot find protoPath` errors each.
  Camera writers can compile shaders after `quick_start` returns; deploy waits
  through this first-use work until `/clock` ticks, or reports a readiness error.
  This does not change the separate driver-wedge detection below.

- Pose publishers use the Isaac 6.1 external transform arrays for USD Xforms
  (including cup-center wrappers), avoiding physics-tensor invalid-target warnings.
  The extension log is capped at 500 model rows and draws visible rows through
  TreeView; callback errors print at most once per source every five seconds.
  Unthrottled loaded-box measurements: 57.14 → 50.21 ms/frame and RTF
  0.2917 → 0.3319; PoseTree warnings 87.5/s → 0. Shared host load varied
  (17.5 → 12.8), so this is an observed improvement, not an isolated CPU benchmark.
- **Sim pacing is a ceiling under shared CPU load.** `up --rate-hz 17.4` (default
  from the recording-rate analysis) sets and reads back Kit's main run-loop
  limiter after the World exists. Each simulation frame advances 1/60 s: target RTF = 0.2900.
  `--rate-hz 19.4` supports the timing sweep; `--rate-hz 0` is unthrottled.
  The launcher disables `syncToPresent`, which otherwise bypasses the main-loop
  limiter. A 65.09 s loaded-box run measured RTF 0.2799, 16.79 Hz and
  59.55 ms/frame (load 23.51); 10 s windows ranged 0.2684–0.2852, so one
  window missed the ±0.02 band. The limiter cannot recover time lost to
  CPU contention. The final 65.19 s GUI handoff run was slower: RTF 0.2672,
  62.38 ms/frame at load 25.02, below the target band. The adjacent unthrottled
  profile at load 20.81 reached RTF 0.5363 / 31.08 ms, leaving 26.39 ms of
  average budget headroom at that measured load. An idempotent `up` reapplies
  the requested ceiling. Existing physics substeps remain 1/120 s; their rate
  is distinct from the observed 1/60 s simulation-frame/clock increment.

- **GPU wedge on NVIDIA driver <580** during cold-cache launches: `up` stops with
  a `GPU wedge (...)` message instead of hanging. See Troubleshooting below.

## Troubleshooting

**Isaac hangs right after the wrist-camera publisher starts, or at `Playing
scene` (first, cold-cache launch).** A known GPU wedge on NVIDIA driver 570.x
(~1 in 4 cold launches in our tests). The launcher detects it (no bridge and no
Kit log progress for 120 s, `ISAACSIM_WEDGE_S`) and `up` stops with `GPU wedge
(...)`, leaving that Isaac running so you can inspect it; nothing is retried
automatically. Stop it with `isaacsim_launch.sh close` and rerun
`deploy_sim.sh up` (idempotent; the caches are warmer now, and warm-cache
launches were 8/8 reliable). The real fix is driver ≥ 580 (Isaac Sim 6.1
recommends 580.95).

**`up` seems stuck at step 1 or 2 on the first launch.** Most likely the one-time
RTX shader compile (~2-7 min; `Waiting for RtPso async group async compilation:
N seconds` in the Isaac log). Let it finish; later launches skip it. Another
Isaac/Kit process on the box can defeat the cache (`Disabling key-value
database because another kit process is locking it`); see the
[NVIDIA forum thread](https://forums.developer.nvidia.com/t/388276).

**`up` stops at step 1 (bridge never answers).** Read the tail of
`/tmp/isaacsim-soarm101.log` (the full Kit log is under
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim Streaming/6.1/`, or `Isaac-Sim Full/6.1`
without `--stream`).
- `Failed to create any GPU devices` (CUDA error 999): `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`.
- `something already answers on bridge port 8767`: another Isaac on the box
  owns that port (9009 is taken too: another project's bridge on the shared
  Isaac install). Use another port for everything:
  `SOARM_KIT_MCP_PORT=8777 deploy_sim.sh up` (the control stack inherits it).

**The Streaming Client can't connect or stays black.** Connect to the exact
address `up` printed. The stream needs TCP 49100 and UDP 47998 reachable
directly (tailnet or LAN); an ssh tunnel can't carry it. One viewer at a time.
`a stream already listens on :49100` means another streaming Isaac is up.

**The arm doesn't move.** Run `deploy_sim.sh status`.
- `up` stops at step 3 with `control_gui needs an X display, but DISPLAY=… isn't
  usable`: you asked for a window (`--gui`/`--rviz`, the default without
  `--stream`) and nobody is logged in to the box's desktop (after a reboot `:0`
  is only the login screen). From an ssh shell `up` looks for the box's session
  itself (the first `/tmp/.X11-unix/X<n>` socket, `:1` on a GDM desktop, plus
  `/run/user/$UID/gdm/Xauthority`). Log in at the box, export
  `DISPLAY`/`XAUTHORITY` for a session you own, or run headless (`--stream`,
  or `--no-gui --no-rviz`).
- Controllers `not all active` in `status`: read the control-stack log it
  prints (the spawners give up after 60 s without a controller manager).
- Policy `not running` or not listening on :8080: read the policy log.
  Common causes are a model that needs a Hugging Face token (`export HF_TOKEN=...`
  before `up`) and CUDA out of memory (check `nvidia-smi`; another job on
  GPU 1, or pick another with `SOARM_POLICY_GPU`).

## Collecting sim data

The normal-only pilot passed at frozen `ef42c1b`: **50/50 accepted**, 17 red /
17 green /16 blue, 30,134 frames at 30 fps. LeRobotDataset load, timing, clean
expert labels, native scene resets and owned cleanup pass. Collection achieved
30.85 accepted episodes/hour (export excluded); median episode RTF was 0.1975.
Bulk data: `/mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v4/normal/dataset`.
Mac evidence: `coursework-eee598/roadmap-2026-11/phase3/3.1-collection/pilot-3.1-v4/`
contains the README, audit, parameters, contact sheet and three colour videos.
Re-grasp episodes are excluded pending their separate physical-clearance review
and smoke. The lead schedules the full run.

Grasp execution uses a yaw-consistent narrow-face plan: TCP centering, projected
width and opening margin are recomputed for each admissible +/-180-degree
orientation. The deliberate recovery search tries collision-valid equivalents
and commands the selected plan's jaws. Normal/GUI jaw commands and actual-jaw
collision checks keep the verified min/max bbox rule and execution route.
It approaches from above inside the open-jaw corridor. Near-block
DART remains nonzero with a smaller clearance-bounded budget and fades for final
contact; receipts report amplitude by phase. Only micrometre support contact at a
**measured start** can be tolerated: targets and swept paths stay strict. Native
fixed-base mode is required, and measured base-link drift is logged separately
from the ROS reference frame. See [descent and support-contact contracts](docs/notes/collection-descent-clearance.md).


The unattended collector shares `PickPlaceNode` with control_gui Quickstart.
`grasp_geometry.py` and the calibrated five-DOF solver in `compute_workspace.py`
are pure Python/numpy, independent of ROS, MoveIt and Tk. Collection uses no
Tk, display, keyboard hooks, spoken announcements or wall-clock lerobot-record.

Physics runs at 120 Hz; application frames and both 640×480 rgb8
ROS2CameraHelper cameras tick at 30 simulated Hz. Controllers, MoveIt and the
expert use Isaac `/clock`. **Observation state is measured PhysX articulation
positions/velocities on `/joint_states_measured`.** `/joint_states` remains the
mock controller reference and is logged separately. It is never the observation.
MoveIt and robot TF also subscribe to physical feedback in the collection preset.

The collector holds the timeline between frames. Given the synchronized image,
physical state and pose snapshot at tick t, it records the clean corrective
label, samples the original planner target for execution, adds seeded DART
offset to that separate array, collision-checks the final motion, and writes
the execution targets into the existing PhysX drives while paused. It reads
those targets back, logs the actual native simulation-clock application stamp,
and advances exactly one frame. A ROS command/acknowledgement protocol applies
each command once. Physical snapshots and motion workers carry immutable
cancellation epochs. One ownership lock covers computation, dispatch, ACK,
recording and completion delivery. Each completion event owns its verdict;
waiting workers revalidate their original epoch after waking. Attach/detach,
scene updates, carried-object bookkeeping and status writes validate and execute
under that same ownership lock, so cancelled workers cannot affect a later episode.
Cancellation/reset waits for ownership within the declared per-query frame
capacity plus a retirement margin; the same capacity bounds capture and motion
watchdogs. A stalled
owner emits a diagnostic and aborts the run; an emergency native cancellation
invalidates queued commands without running reset setters. Its RPC is separately
bounded at 30 wall seconds. A poisoned run cannot install another generation. Reset invalidates every observation at or before its native sim stamp,
even if a delayed message carries the new epoch. The audit requires observation,
expert and application epochs to agree throughout an episode. Approach/descent targets,
completion identity, timing, tag and planner settings live in a frozen
`MotionOperation` passed directly between continuations. Worker thread context
carries its event; the UI's current-event slot is not a continuation input.
Planning/validation runs outside generation ownership. Only short state/command
commits and scene mutations acquire the guard, then validate the original request.
Capture, scene and cancellation timeouts share the setter-free emergency abort
and structured diagnostic. See [operation ownership](docs/notes/collection-operation-ownership.md)
for the shared-state sweep and concurrency contract.

Scene service completion is checked separately from local ownership. An unresolved
attach/detach, world-object or ACM mutation aborts and poisons collection; late
completion cannot authorize another reset. A persistent runtime marker requires
a new instance control process before a later collector starts. A timed-out Kit
pose/reset RPC requires fresh control and Isaac processes. Use this instance's
`collect_sim.sh down N` / `up N`; the marker retains the diagnosis.

The marker error names its path, original request error and missing/unchanged
process identities. Marker reads and temporary-file creates use no-follow
opens; temporary files are exclusive, flushed and atomically renamed. If an
identity was unknown when recorded, **a restart cannot clear the marker**.
Malformed/unreadable markers also refuse startup.

**Manual scene reconciliation:** stop only the assigned instance with
`collect_sim.sh down N`, confirm its recorded control/Isaac process groups and
MoveIt server have exited, and start fresh control and Isaac processes with
`collect_sim.sh up N`. Before collecting, an operator must inspect the full
MoveIt scene (`GetPlanningScene`, mask 1023): no attachments, unexpected world
objects, occupancy or stale collision exemptions, and the intended fresh
physical scene on the assigned domain/bridge. Save those queries and process
identities with the original marker under this instance's runtime directory.
Only after that inspection, rename the explicit `scene-unresolved.json` leaf
to an explicit diagnostic filename in the same directory. The collector never
automates that override. If the old processes or scene cannot be verified,
leave the marker and collection stopped. Inspect leaf symlinks without opening
their targets and retain malformed-marker bytes for diagnosis.

Every reset queries the full MoveIt scene, removes remote attachments independently
of local bookkeeping, clears world objects and rebuilds collision allowances from
the checked-in SRDF. At the empty-world barrier the policy contains only robot
pairs. Base/ground and object/ground allowances are added only when the rebuilt
world contains `ground_plane`; removed objects cannot retain permissions.
Final verification requires exact expected world IDs/poses,
no attached objects, the correct ground, measured robot state and canonical ACM.
ACM comparison checks off-diagonal pairs: missing names/pairs mean false and
self-pairs are irrelevant. All robot/world permissions remain strict, including
rejection of defaults. The CPU service fake models MoveIt 2.5.10's
[constructor](https://github.com/moveit/moveit2/blob/2.5.10/moveit_core/collision_detection/src/collision_matrix.cpp#L66-L96)
and [serializer](https://github.com/moveit/moveit2/blob/2.5.10/moveit_core/collision_detection/src/collision_matrix.cpp#L332-L361),
plus [world-removal pruning after ACM copy](https://github.com/moveit/moveit2/blob/2.5.10/moveit_core/planning_scene/src/planning_scene.cpp#L329-L347).
Final verification checks each rebuilt primitive's type/dimensions and each
mesh's vertices/triangle identity against the shared shape builders, with
1e-7 m numeric tolerance and harmless mesh indexing/order changes accepted.
Both cleanup and final rebuild require empty occupancy. The first startup reset
writes `first-reset-scene.json` in the run directory, including the native ACM,
world/attached IDs, occupancy size and geometry checks. Accepted receipts retain
the same cleanup and verification results. Include this file in the smoke report;
CPU serialization tests do not establish the live MoveIt round trip.

Scene policy is immutable and validated at node construction: collection requires
its effect, scene-failure and callback-failure hooks together. The GUI uses raw
MoveIt scene services and its established background dispatch; it has no scene
worker, resync mechanism or collector completion barrier. GUI scene updates are
not transactional (tracked as an Ops item in ROADMAP). Collector scene requests
remain strictly tracked through completion, including unresolved-request refusal
and authoritative reset reconciliation. Collector target removal is a completion
barrier before validation, and its Gate C traces retain native contact pairs.

Subscription, timer, service and action-result callbacks contain exceptions at
the ROS boundary. GUI exceptions are logged without killing the executor. Every
collector callback abort poisons its local generation before the native emergency
pause. Capture failures retain the original exception, phase, epoch, measured
state/reference and cancellation result in the episode receipt; an unrelated
clock/reset/scene/transport failure keeps its own error.

The collector preserves an **idle** clean hold target through a cancellation/reset
barrier and tags it with the newly installed epoch. Cancelling active motion
retires its target. Before and after scene reset it explicitly holds the nominal
calibrated home and checks a six-tick measured window, allowing at most five sim
seconds to settle. Home motions/holds receive no DART noise. The existing 0.01 rad
home error, 0.002 rad spread and 0.008 rad motion-convergence limits remain.
After a quality rejection, the next attempt first restores the saved, verified
home snapshot, including all six physical joint positions and drive targets and
rigid-body poses/velocities. The reset retires workers and clears the authoritative MoveIt scene, then
restores the native snapshot, verifies readback and six contract-checked settling
ticks, and rebuilds the synchronized scene before planning. A failed restore is terminal; the collector never
plans a home trajectory out of the previous failed contact pose. The baseline's
calibration, samples and snapshot hash are retained in `run.json`.

Collector reset rebuilds include **every** physical lego, including one still
resting in a cup before the next randomization. `SceneReconciler.rebuild_legos`
uses the shared geometry factory and the synchronized native snapshot; full
world/pose/shape/ACM verification stays strict. GUI scene refresh retains its
verified in-cup filter for post-drop home planning. The distinction matters at
`reset_release`, which runs before the next seed's randomization.

The sensor join retains raw stamps, layout/content hashes and each packet's RMW
publisher identity. Camera pixels are provisional until they match exactly one
native physical tick within the shared 1 us producer tolerance. It permanently
keeps the **first taken image** for that tick/publisher/layout, including after
consumption; a held re-render may have a slightly different raw header stamp.
Identical packets count separately from changed-pixel or changed-header re-renders.
Different GIDs/layouts, no matching native tick or multiple matching native ticks
are terminal. Selection proof records the copied native epoch/tick and unchanged
selected camera stamp; export requires this proof. Pending measured joint states
cannot overwrite a different same-tick state. Camera packets at/below the
cancellation floor plus the producer allowance, and pre-epoch bootstrap packets,
are discarded before source binding/ambiguity checks.

Kit can re-render while physics is held, so RGB variation alone does not imply a
second publisher. Startup and each episode boundary require one graph endpoint
per camera and retain its GUID and QoS; every image also carries packet identity.
One-second graph checks alone are insufficient: the checked-in real
ROS counterexample delivered a first same-format frame from a transient second
publisher inside 0.201 s while startup/1 s/2 s checks all saw one source. Humble's
Python API omits packet GID; the collector uses an isolated project-local native
serialized take on the existing borrowed subscription handle, then normal Python
ROS image deserialization. No system ROS or publisher proxy is installed. A C
compiler and Humble development headers/libraries are required; the helper builds
only against the checked-in, validated Humble ABI manifest: transitive headers,
loaded RCL/RMW/CycloneDDS/rclpy libraries and executor implementation. Unknown
inputs or a cached helper differing from its build receipt refuse startup.
Source/library hashes and warm decode costs are recorded. Its executor adapter handles only the
two marked image subscriptions; other callbacks retain Humble behavior. Existing
callback containment turns transport errors into terminal run evidence.

CycloneDDS's received-message GID is a subscription/context-local publication
handle, not its graph endpoint GUID. Metadata records that scope; IDs from the
two namespaces are not falsely compared. This identifies packets in this
receiving subscription and context. See the
[transport contract and real ROS proofs](docs/notes/collection-image-identity.md).
Render-product pausing is a separate performance experiment, not required for
correct first-frame selection.

The epoch and cancellation floor are published as one identity. Camera stale
filtering and publisher binding share that identity lock; durable episode receipts
allow fenced image/re-render history to be pruned. The shared node explicitly
destroys its ActionClient waitables before base-node destruction. Collector and
clock checks retire the spin thread and callback pool before destroying nodes and
contexts. Clock checks require an invocation-bound post-cleanup certificate and
refuse abnormal producer exit. Capture retirement and executor retirement must both pass
before native node/context destruction. Async profiles keep copied native IDs and
selected camera headers in separate fields. Attempt finalization preserves the
original terminal fault and publishes its seed/poses, raw profiles and camera
counts even if supplementary profiling or terminal-marker publication fails;
the failure streak observes the final disposition once. Unresolved camera groups
refuse an episode boundary after a bounded native-publication drain. A sealed
boundary also rejects late unmatched headers in the gap before the next valid
physical tick; camera-first packets for future ticks remain provisional. See [ROS lifetime evidence](docs/notes/collection-ros-lifecycle.md).

A typed physical convergence deadline can reject one episode after native pause,
worker retirement and contract auditing. Stationary/short data, calibrated-home
instability, explicit planner collision refusal, failed containment and insufficient
lift are logged quality rejections. Camera/state joins, ACKs, clock/reset/epoch,
transport, scene, unknown failures and mixed audit failures stop the run. Three
consecutive rejections in one reason category also stop it; receipts and summary
persist that decision and resume refuses a stopped ledger. Runs carry a versioned
quality policy; use a fresh output directory for this policy rather than migrating
an older pilot ledger. Failed episodes remain excluded from export.

Smooth DART uses seeded minimum-jerk ramps with 0.3–0.8 s holds, nominal
0.26–1 degree per-joint amplitudes and a 65% per-arm-operation on probability.
The last 20% is quiet. Short operations scale amplitude only when both the
0.3 s minimum hold and 0.35 s minimum ramps fit; otherwise noise is skipped.
Home and gripper motions receive no injected noise. The original planner time
law and waypoints are sampled directly. Execution adds an offset to that
reference; it never filters, independently clips or retimes the expert path.
The clean recorded label remains exactly
`corrective_action(measured_state, expert_reference)`.

Before an arm operation starts, native MoveIt checks its complete execution
path with the proposed noise. It chooses a fixed scale (1, 0.5, 0.25 or 0) for
that whole operation, preserving smooth ramps, and logs rejected candidates and
contacts. Zero-noise failure refuses the operation. This planning work runs
outside generation ownership. Immediately before every dispatch, MoveIt checks
the final six-joint target and both short continuations (measured state to target
and previously applied target to target), including gripper and native attached
geometry. Both endpoints and subdivisions at most 0.01 rad apart are checked.
Final checks bound individual validity requests at 1 s (50 ms polling/delivery
allowance) and cap a continuation at 256 queries; operation preflight caps total
queries at 50,000. There is no aggregate five-second deadline on valid answers.
Actual per-tick query counts, total time and maximum request time are logged. An unexpected invalid continuation
or query fault aborts while physics is held, before publication/advancement. It
never jumps to a differently filtered expert command to avoid a collision.

Frames log `planner_execution_reference`, scaled `dart_offset`, geometry checks
and actual target velocity. `planned_noise_offset` is the total execution-minus-
label difference; `command_filter_delta` is zero. The audit independently checks
planner-plus-offset identity, clean labels and geometry proof identity. The
noise itself is bounded at 0.05 rad/s and 0.30 rad/s². Overall target derivatives
inherit the original planner profile and are reported rather than reshaped;
legacy linear profiles can have sampled acceleration spikes at motion boundaries.
Reproduce the offline command comparison with
`dart_benchmark.py --input <retained-run> --output <confined-dir>`.
It preserves original frame timestamps; it is not a physical yield test.

Every joined physical tick, including unrecorded home/settling ticks and a failing
convergence tick, passes the same source/shape/finiteness, camera layout, clock,
epoch, one-frame progression and previous-ACK persistence contract. The collector
waits up to two wall seconds for ROS `/clock` to join the held native tick.
Application/ACK and clean-label checks finish the tick's receipt. A convergence
failure first obtains a no-advance hold ACK and saves that failing tick's own
verified receipt before cancellation. Physical verification snapshots carry
row hashes and their checked provenance. Native camera/joint/ROS stamps keep
strict same-tick joins. `World.current_time` independently accumulates PhysX
callback step sizes; it is retained as measured, without requiring absolute
equality to the graph/camera clock. Each clock must advance one frame. World
increments use a tolerance of max(1 microsecond, four float32 timestep errors
plus eight double-precision ULPs), bounded below a quarter physics substep;
native stamp progression remains within 10 ns. This accepts accumulated clock
drift while rejecting frozen, rewound or skipped physics frames.

Target-preserving pauses and physical restores have explicit epoch-stamped
baselines. Restore reads back all six physical drive targets and the native/World
clock pair; Servo history and the tick audit adopt that verified baseline while
preserving the clean home reference. The next tick must match restored targets
before dispatch. Command velocity provenance uses current physical target
read-back. Missing/mismatched restore identity, target overwrite or clock rewind
is terminal. Contract failures are terminal;
physical home/containment failures are separate verdicts and may be quality
rejects only with valid tick evidence. The six post-velocity-reset home samples
come from the new epoch; samples are never mixed across a reset.

Re-grasp recovery occupies 18% of **accepted slots**, evenly spread through
colour rotation (9/50; 162/900, or 54 per colour). Retrying a rejected slot keeps
its recovery type and colour but logs a new seed. `--regrasp-only` makes every
slot a recovery for the dedicated smoke.

Recovery starts from a **synthetic post-miss reset**. Seeded requests sample
26–38 mm lateral TCP offsets around the true grasp, small yaw deviations (up to 12°),
and three jaw phenotypes: 50% closed, 30% partly closed, 20% open with a higher
hover. The sampler retains its phenotype while trying up to 24 reachable poses.
Receipts record the requested distribution, selected pose/jaw, every refused
candidate and its MoveIt contact pairs. Accepted proportions can differ from
the requested mix and must be reported.

Before writing any native state, the calibrated shared IK and joint limits,
MoveIt full-scene validity **with the target present**, and independent native
robot-box checks must all pass. The native model covers every enabled robot
collision box. Robot boxes require 1.02 mm clearance from all dynamic block/cup
primitives; finger boxes also require that clearance from the floor. The 1 mm
margin covers the measured 0.1 mm finger + 0.1 mm target contact offsets, with
20 um arithmetic allowance. MoveIt retains self-collision and ground policy.
Finger/block effective offsets are freshly read and verified per episode.

The existing held native restore sets all six named joint positions, matching
drive targets and zero velocity, with immediate readback and epoch/command-history
rebasing. Six unrecorded normal-dynamics hold ticks then verify the complete
pipeline contract and static geometry. A native guard measures **every cup and
block**, including neighbours and the target, against the pre-restore world
anchor: corner motion <=0.2 mm, linear speed <=0.7 mm/s, angular speed <=0.012
rad/s. It samples all four 120 Hz pre-physics callbacks plus the final post-solve
boundary in each tick; a transient excursion or missing sample is terminal.
Both physical base frames must retain their held float32-readback poses. The
writer starts only after all 24 callback samples, six final boundaries, scene
synchronization and independent certificate reconstruction pass. These velocity
limits are empirical contact/impulse screens, not solver displacement bounds:
[rest-velocity calibration](linux-env/scripts/collect/fixtures/rest-velocity-floor.json)
measured pre-contact maxima0.34691 mm/s and0.0056004 rad/s. The screens provide
about2x margin; the0.2 mm per-substep displacement gate is unchanged. Every
receipt reports native per-body substep peak velocities. Any resting-body peak
above half a velocity limit must be reported before production. The shared Quickstart then
performs the correction at normal speed and dynamics. Setup failures are terminal
with durable native evidence; exhausted collision-clear candidates are planner
quality refusals. Receipt replay restores the saved synthetic world anchor
(all body poses) and selected joints, with zero velocities and matching drives;
it refuses changed fixed bases, body geometry or the saved robot geometry/kinematics before writing. Its certificate
binds world/base readback, clocks, drive baselines, all substeps and the first
corrective tick. The lateral offset excludes the separately sampled hover, so
the 3-D displacement can exceed 38 mm.

Every exported frame carries int64 `collection_phase` (shape `[1]`): **0 normal,
1 miss_prefix, 2 corrective**. All frames from synthetic-reset recovery are
**2 corrective**, with clean expert actions recomputed from measured state and
ordinary smooth DART applied only to execution. Setup/reset/settle frames are
unrecorded. Train on normal and corrective frames; the phase column is selection
metadata, not a deployed policy input. The exporter retains support for historical
phase flags and refuses setup frames or a physical-prefix receipt under the
current synthetic-start policy.

These episodes are state augmentation, labelled **synthetic post-miss resets**;
they do not claim a physically observed failed grasp. They target the established
single-observation policy. A policy conditioned on motion history would require
reconsidering the reset distribution. The unchanged final lift/containment,
timing/command/epoch audits and success-only retention apply. See
[recovery start and evidence contracts](docs/notes/regrasp-target-clearance.md).

Frames and receipts carry `recovery: regrasp`; datasets retain the full
per-episode request, geometry, restore and checked settle evidence in
`meta/collection_episodes.json`. Acceptance, resume and export revalidate the
synthetic-start certificate against the corrective rows and phase boundary.
No native velocity limits or gravity settings change during synthetic setup.

This source cut also changes NORMAL collection relative to the frozen
`bf7336d`: NORMAL exports gain `collection_phase=0`, and runtime startup,
cancellation and per-frame pauses use a committed-pause barrier. The shared
GUI/expert motion module is unchanged; runtime and export qualification does
not transfer from that byte identity. NORMAL batches on this cut need their
own startup/clock and end-to-end qualification; production remains on its
separately validated frozen cut.

Smooth DART and re-grasp require a validated smoke at the current scene/cut; their physical yield and
side-by-side/recovery videos require targeted review and a newly authorized
smoke before a fresh pilot. The stopped pilot-v2 ledger remains diagnostic.

The CPU GUI parity tests pin 14 scene/wait/dispatch methods to the live-verified
`6e439ed` implementation and compare service sequences for normal and delayed
responses. An offline ROS regression boots the actual GUI node against concurrent,
delayed fake MoveIt services, injects callback failures and checks executor/service
liveness. It launches no Isaac instance. Add `--tk` with a usable X display to also
exercise the actual Tk Refresh/detach/log path and event-loop heartbeat:

```bash
source /opt/ros/humble/setup.bash
source vla_SO-ARM101/install/setup.bash
export ROS_DOMAIN_ID=95 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SOARM_INSTANCE_ROOT=/mnt/a4500-backup/exploring-vlas/runtime/instance-5
export CYCLONEDDS_URI="file://$PWD/linux-env/cyclonedds.xml"
unset ROS_LOCALHOST_ONLY
PYTHONDONTWRITEBYTECODE=1 python3 -B linux-env/scripts/collect/gui_scene_regression.py \
  --instance 5 --output /mnt/a4500-backup/exploring-vlas/runtime/instance-5/gui-async-scene-regression.json
```
Use an unused test domain/runtime. Both ROS startup probes accept `--instance N`
and require domain `90 + N`; they never start Isaac. `--normal-only` excludes all
re-grasp branches, and resume/replay/export reject a mode mismatch.
Pose-cache subscribers recheck bypass under the map lock;
controller-reference diagnostics carry source time/epoch validity and are never
used as expert labels. Post-step
drive-target read-back checks that the command persisted through the next frame.
Only this instance's normal ROS follower actuation edges are disconnected during collection. Shared planning, grasp/IK maths,
attach/detach and drop logic remain the GUI implementation. Planning/service
work runs with the sim held; verification tail ticks are excluded from training.
The audit rejects skipped/duplicate ticks, mismatched physical state/images/poses,
noisy labels, changed applied targets, and application latency above one tick.
A five-sim-second-late application probe must fail.

Success and collector MoveIt cups share
`isaac/exts/soarm101-dt/assets/sorting_collision_geometry.json`: twelve tapered
wall boxes and one floor box per cup, in the cup-base frame. The `/drop_poses`
centre is 48.26 mm above that base along the cup's local Z. MoveIt receives all
thirteen boxes with their exact dimensions and wxyz orientations, and reset
verification matches each shape together with its world pose using a one-to-one
geometric match. The GUI scene path retains its established behavior.

All eight block corners must satisfy all twelve interior planes. Asset normals
are not unit vectors: signed clearance is `(offset - n·corner)/||n||`. Floor top
is 1.2 mm; the rim is 96.52 mm. Wall/floor contact permits at most **20 um** of
penetration (9 um measured rest contact + 10 um solver variability + 1 um float
allowance), below the 0.1 mm contact distance; rest offsets are zero. The upper
gate is 0.1 mm below the rim. There is no STL radial envelope or 0.7 mm inward
margin. The 15 settled ticks and 25 mm lift requirement remain. Metadata records
the primitive asset hash, per-corner plane distances and authored expected
offsets/counts. Fresh native tensor reads must match those expectations at each
collector installation; old measured offsets are never relabelled as current. See [cup geometry](docs/notes/collection-success-geometry.md).

The collection World is created with `device='cpu'`; the loaded scene explicitly
disables GPU dynamics. Runtime refuses an absent/unknown/true GPU-dynamics value
and records the verified CPU scene setting in run transport metadata and each
episode receipt. CUDA remains pinned for rendering/other CUDA consumers. World
timing and physics mode are launch-time properties, never changed on a live scene.

For smoke/diagnostic runs, `--diagnostic` retains failed-attempt videos and saves
`final-top.png` / `final-wrist.png` from the final recorded frame without advancing
simulation. The receipt records its frame index and native sim stamp; the settled
verification tail is outside the video. Production defaults still remove failed
videos. Both modes exclude failed episodes from LeRobot export. Retention mode
must stay the same when resuming a run.

The collector installs simulation-clock motion waits; the GUI retains its
wall-time waits. Each event owns its planned duration and timing origin. Linked
approach/descent budgets include both stages; the actual trajectory duration
updates the budget before stepping. A motion gets its planned sim duration plus
5 sim seconds for convergence. The separate wall watchdog is at least the
legacy bound or `(sim budget / --wait-rtf-floor) + 30 s`, whichever is larger.
The default RTF floor is 0.1, allowing the tested 0.27 and 0.15 runs. Whole-cycle
budgets grow with dispatched trajectories/gripper moves; `--episode-deadline`
is the minimum wall watchdog, not a fixed robot-motion deadline. Expiry logs its
kind, epoch, planned/elapsed sim time and wall limit. RPCs, ACKs, frame joins and
ownership/retirement retain wall deadlines. Wait policy is logged in `run.json`
and must match on resume. Partial backward clock jumps are detected against the
previous sample, even if still above the wait origin. `clock-sanity` installs
the same sim-time waiter for its FJT result; its duration check logs a delivery
allowance of one 30 Hz clock tick plus the 50 ms wall polling quantum scaled by
measured RTF, in addition to the 0.4 sim-second controller/transport allowance.
See the [complete wait sweep](docs/notes/collection-wait-clocks.md).

`deploy-clock` restores the ROS follower, keeps physical feedback publishing,
and caps the identical simulation/camera profile at 30 app Hz (target RTF 1.0).
`deploy` also starts the policy launcher with **fps=30**, action duration=1/30 s
and `--robot.joint_states_topic=/joint_states_measured`. The existing LeRobot
robot config already supports this topic option; real hardware keeps its measured
`/joint_states` default. A cap does not guarantee throughput: measure actual RTF
before accepting deployment. Legacy eval defaults remain separate from this preset.

Instance N selects domain `90+N`, bridge `8777+2N`, HTTP `8778+2N`, policy RPC
`18090+N` and private runtime files. `ISAACSIM_GPU` selects renderer, physics
CUDA and CUDA consumers (default 0). Indexed launches resolve that physical
card's UUID and pin `CUDA_VISIBLE_DEVICES` to it. CUDA consumers use its remapped
ordinal 0. Renderer indices come from Kit's GPU table, resolved against UUID and
PCI bus identity; they are independent of nvidia-smi indices. Startup and per-attempt
GPU receipts include both NVML compute and graphics process lists on every card.
Same-PID/card memory is not counted twice. Missing API coverage or memory
attribution is **unverified**, never zero. Birth identities and the owned descendant
set are checked before and after sampling; unstable samples get three bounded
attempts and then refuse as unverified. Any verified owned foreign-card context
stops collection. A prior owned Kit log supplies the launch-index hint; startup
requires the current log's sole Active device to match the assigned UUID and PCI
bus, and collection audits it again. A fresh instance without a prior device table
refuses before launch; provision its owned table during an explicitly assigned
setup window. These checks verify sampled allocations; they do not create an OS
device-access boundary.
Use only assigned resources on shared boxes. DDS uses loopback-only CycloneDDS;
do not also set ROS_LOCALHOST_ONLY. World timing is set before scene creation,
including the 30 Hz frame-rate clamp. Never change World timing on a live scene.

All run/export/runtime paths resolve beneath the canonical
`/mnt/a4500-backup/exploring-vlas/`; environment overrides stay within it.
ROS logs, temp, HF/OptiX caches, portable roots and per-instance grasp traces
are directed there. The indexed launcher overrides inherited
`MCP_CLIENT_OUTPUT_DIR` with its confined instance resources directory and validates
the resolved path before creating it. Direct Python entrypoints apply confinement before imports that
create trace/log/cache files. Indexed launches use persistent per-box/per-card
shader, OptiX and CUDA caches beneath `cache/isaac/<box>/<GPU-UUID>/`; inherited
cache paths are overridden. Existing `~/.cache/ov` contents are never cleared.
Bring-up warms both cameras for 30 unique frames before collection and writes
`cache-warmup.json`. The default Kit task pool is six threads (configurable 1–6)
and physics uses four threads per instance. These are capacity limits, not a claim
of measured multi-instance throughput. Require
50 GiB on the output filesystem before each attempt; below that, wait every
60 s until 52 GiB is available. Root has a **20 GiB last-resort stop**, including
launcher startup: the earlier 90 GiB rule protected root writes, while collection
bulk data lives on another filesystem. Never alter the sibling backup archive.
Receipts measure growth of writable root files in this instance's process trees,
identifying shared caches. This is space growth, not in-place rewrite traffic.

Every capture includes cheap wall timing in an immutable `wall_profile`; receipts retain all
attempt ticks plus mean/p95 for Kit step-to-post-update, camera arrival/join,
in-process native state reads, MoveIt validity query count/time, expert/DART
compute, publish-to-ACK, contract checks, image decode and row/encoder writes.
The common clock is same-host monotonic wall time, never a dataset timestamp.
Kit post-update is a render-completion proxy rather than a GPU fence; the separate
step-to-both-camera-arrivals interval includes ROS transport. Async video writes
can overlap another tick, so stage shares are reported against total tick time
without claiming an additive partition. Encoder completion is persisted in
`write-profiles.jsonl`, linked by profile ID, epoch and sim stamp; it never changes
an already queued capture row. Receipts retain separate capture/write records and
join them for statistics. Failed joins, contracts, geometry or ACKs retain partial
profiles with their failing stage, including final shutdown in
`wall-profile-finalization.json`. Per-tick native reads use no bridge RPC.
The next live smoke must measure these costs; offline checks do not establish
throughput.

The combined smoke runner uses the same launcher, collector and export entrypoints.
It first inspects one attempt in each mode, then resumes to five accepted normal
and five dedicated re-grasp episodes, retaining failed videos with `--diagnostic`.
It saves first-reset/restore receipts, stage/attempt wall times, dataset-load
validation and PID/start-time/pidfd cleanup evidence. Receipt scans reconcile once
more after each producer exits. Every JSON publication is atomic; after exit an
invalid final receipt fails. Cleanup/probe errors still publish a failed smoke
receipt and release the instance claim. Terminal markers publish before cleanup
and are never cleared; any root/attempt terminal marker forbids resume/export or
success. Bring-up of a collector invocation invalidates earlier completed summary
status. Its finalization certificate must affirm cleanup, match the producing PID
and stage time, and hash the final summary; a stale accepted/complete receipt
cannot substitute for it. Resume validates the predecessor before constructing
or beginning a replacement collector; incomplete/missing certification refuses.
A valid predecessor certificate and exact summary are archived before the new
invocation changes status. Cleanup for an invocation that never began writes
separate instance runtime diagnostics, preserving refused predecessor evidence.
Executor shutdown must return True and its spin thread must retire; timeout or
unknown completion is terminal. Certificates check schema and matching invocation
as local consistency evidence; they do not authenticate a same-permission writer. A shutdown exit code alone never proves
or disproves a completed collection: accepted episodes and export validation
receipts are required, and cleanup must independently verify every owned child. `--plan` starts no processes. `--go`
requires the lead's explicit window; independent review can run concurrently,
and its verdict can supersede the smoke data. Visual review and pilot approval
remain separate from collection completion:

```bash
python3 -B linux-env/scripts/collect/run_smoke.py --cut <full-commit> \
  --output /mnt/a4500-backup/exploring-vlas/datasets/smoke-clock-restore-v5 --plan
# In the granted window, use the identical command with --go instead of --plan.
```

The runner and indexed launchers share an exclusive `instance.lock`. The runner
tracks kernel descendants (including orphaned grandchildren through a child
subreaper), never adopts PID-file contents, and signals only its recorded pidfds
after checking each identity again. The indexed Isaac log is pinned to
`runtime/instance-N/isaac.log`. A startup failure without an attempt receipt keeps
the original traceback as the primary smoke error and counts zero attempts.

Constructor status initialization has a separate synchronous bootstrap owner;
this owner is revoked before the ROS/capture threads start. No motion or callback
may use `epoch=None`. Cancellation epochs must strictly increase, and both clocks
are reconciled against the latest row or held boundary even across repeated
pause/restore calls. The held-frame capacity is 325.8 wall seconds (256 queries
at up to 1.05 s each, plus 57 s for ACK, clock delivery, recording and scheduling);
retirement allows another 5 s. Collector motion waits add this frame capacity to
their 30 s wall margin, leaving a full valid-query frame budget before the physical
motion wall watchdog expires. These are hang limits, not pacing targets. The GUI waiter is unchanged.

All cross-producer numeric agreement goes through `comparisons.py`; copied
fields and proof hashes remain exact. The full inventory and retained-snapshot
regression are in [collection-clock-comparisons.md](docs/notes/collection-clock-comparisons.md).
Failed-verdict receipts retain both values, deltas, rules, snapshots and proofs.

The launchers use [managed_process.py](linux-env/scripts/collect/managed_process.py)
to supervise only their own process trees. PID files now name that supervisor;
identity receipts retain workload and child PID/start-time/PGID. Close, restart,
collection down and deployment/stack down verify ownership and claims before
requesting pidfd-only child cleanup. A legacy PID-only file grants no shutdown
authority: a live foreign or reused PID is refused, never adopted. See
[launcher-process-ownership.md](docs/notes/launcher-process-ownership.md).

Recording provenance inventories the local Python sources loaded by the recorder
and the native Kit runtime, including dependencies imported only by Kit. It
retains initial fingerprints and refuses missing or changed files and local
modules resolved from another checkout. Asset/config files, native entrypoints
and image ABI inputs are included explicitly. The comparison runs before the
first collection tick; the real-node CPU probe exercises it before reaching its
held-tick checkpoint. No per-tick source scan is added.

The integration checks below use fake Kit and MoveIt/ROS message sources and
start no simulator. Run the real-node check on the isolated assigned domain with
system Humble and the control workspace sourced. The full attempt uses ideal
bounded fake physics, collision-free fake MoveIt answers and synthetic camera
pixels; its artifacts are explicitly test-only. It proves the real collector's
construction, provenance, episode and finalization paths, while physical
clearance, solver residual and visual recovery still require the live smoke:

```bash
# Cyclone XML already selects lo; do not also ask the RMW to add it.
unset ROS_LOCALHOST_ONLY
ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  CYCLONEDDS_URI="file://$PWD/linux-env/cyclonedds.xml" CUDA_VISIBLE_DEVICES='' \
  python3 -B linux-env/scripts/collect/collector_boot_regression.py --instance 9 \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-9/bootstrap-check

# Full fake attempt: synthetic reset, settle, shared corrective cycle and finalization.
# Use fresh output paths. The late-fault variant proves durable failure finalization.
ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  CYCLONEDDS_URI="file://$PWD/linux-env/cyclonedds.xml" CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B linux-env/scripts/collect/collector_boot_regression.py --instance 9 \
  --full-regrasp --output /mnt/a4500-backup/exploring-vlas/runtime/instance-9/regrasp-check
# Repeat separately with --inject-native-fault, --inject-restore-failure,
# --inject-moving-target, --inject-moving-cup, --inject-moving-neighbour or
# --inject-transient-neighbour and fresh output paths to test terminal cleanup.
# --replay-synthetic-receipt <successful-run>/attempt-0000/receipt.json checks
# exact saved-world/joint replay through the same complete attempt.
python3 -B linux-env/scripts/collect/process_ownership_regression.py \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-0/process-ownership-check
```

```bash
# Live commands require the lead's explicit smoke/pilot window.
# Use the box/domain/GPU assigned by the lead. Instance 0 here is GPU 0/domain 90.
ISAACSIM_GPU=0 bash linux-env/scripts/collect_sim.sh up 0

bash linux-env/scripts/collect_sim.sh clock-sanity 0 \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-0/clock-sanity.json

bash linux-env/scripts/collect_sim.sh run 0 --episodes 5 --seed 330000 --diagnostic --regrasp-only \
  --output /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-smooth-regrasp-v1
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-smooth-regrasp-v1 \
  --repo-id local/soarm101_headless_smoke \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-smooth-regrasp-v1/report

# Normal-only pilot: explicit mode disables the mixed recovery quota; resume requires the same mode.
# Accepted colours rotate 17/17/16. Recovery smoke and later recovery collection are separate.
bash linux-env/scripts/collect_sim.sh run 0 --episodes 50 --seed 350000 --normal-only \
  --output /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v4
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v4 \
  --repo-id local/soarm101_headless_pilot \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v4/report

# Optional: policy deployment using the same measured state and 30 Hz preset.
bash linux-env/scripts/collect_sim.sh deploy 0 --pretrained /path/to/model
# Stop only this instance/control stack, after collection/deployment exits.
bash linux-env/scripts/collect_sim.sh down 0
```

`--resume` re-audits an interrupted run's receipt ledger and continues its original
baseline, next seed and accepted-colour rotation; inconsistent ledgers and changed
motion/timing sources are refused. `--replay-receipt /path/to/receipt.json` restores
the recorded initial physical body poses, orientations and zero velocities,
plus the measured arm state and clean expert reference. Normal collection and
replay both advance two unrecorded reset ticks to refresh measured state and TF;
receipts retain the reset state and the state at the start of execution. Missing cups cause a
clear refusal requiring a fresh scene. Collection never adds/deletes rigid bodies
beside the articulation. Each reset clears linear/angular velocity for cups and
blocks and verifies six measured home position samples before zeroing initial
velocities. Raw joint velocity samples are retained; constraint impulses are not
relabeled as zero motion. Replay restores poses exactly within
1 μm/1e-6 quaternion tolerance. Immediate held-clock read-back also requires all
six arm positions within 1e-6 rad of the receipt and all six velocities within
1e-6 rad/s of zero. The later physical settling samples retain constraint impulses.
Before planning, target pose caches come from the joined physical reset snapshot;
MoveIt must return matching cup/block positions and orientations, including the
oriented cup body-center/mesh-base offset, plus measured robot state.
Contact and planner execution need not be bit-identical.

`run.json` records clock/transport settings, source hashes and held-out eval seeds
900000–900029 (forbidden in training). Receipts retain every attempt's task, seed,
initial physical poses, cup layout, approach/speed/noise parameters and outcome.
Only successful audited episodes enter the dataset. Success requires all eight
corners of the selected block's oriented bounding box inside a conservative cup
interior in the cup's local frame, below the rim, with low block/cup velocities
for 15 consecutive sim ticks. The block must also have lifted at least 25 mm.
The same primitive half-planes used by MoveIt bound the oriented block in the
cup-base frame. Both floor and wall checks admit the documented 20 um rest
residual; the rim has a 0.1 mm clearance. Floor-supported wall-contact placements
pass, while meaningful side protrusion, below-floor and rim placements fail.
Smoke must confirm these gates against physical boxes and recorded video;
offline geometry proves the contract, not live placement yield.

State/action use absolute degrees in deployment's six-joint order, including
`gripper.pos`; raw audit rows retain radians. The action is the clean corrective
expert command. Seeded disturbances affect only `executed_action`; actual PhysX
read-back targets are logged separately. Minimum-jerk drift segments use nominal
0.26–1 degree offsets, 0.3–0.8 s holds and at least 0.35 s ramps, with 65% per-arm-
operation probability. Home/gripper operations and the final 20% stay noise-free.
The recorded noise operation includes its seed, full program and geometry scale. Diversity also covers object poses, cup line/arc
layouts and colour order, approach offsets/heights and speed.

`summary.json` must have complete=true and COLLECTION_FINISHED; exit code alone
is insufficient. In-progress checkpoints are explicitly marked. Episode/step
wall deadlines fail bad attempts. Failed receipts remain; failed videos are
excluded. The exporter refuses partial/superseded runs, re-audits original ticks,
checks exact row/video counts, writes measured relative sim timestamps through
the pinned LeRobotDataset API, reloads the dataset, and creates a contact sheet,
three colour sample videos and parameter/timing reports. A completed export can
be checked again with the export command's `--validate-existing` flag: it refuses
missing or different episode/phase metadata, re-audits the raw rows and videos,
and compares every stored state, action and phase with that source without
rewriting the dataset. Every stored timestamp, frame/episode/global/task index and task assignment is checked against raw. Both camera videos are decoded and checked at EVERY frame for count, PTS order and raw content: lossless RGB must match exactly; H264 requires per-frame PSNR>=32 dB and max absolute RGB error<=224. Those lossy bounds accommodate measured chroma-edge re-encoding error; they do not claim pixel identity. A failed audit invalidates the previous validation certificate. Standalone export resolves the shared pure IK package
without requiring a sourced ROS workspace. Report accepted yield,
RTF, idle/stationary fractions, storage growth and episodes per wall-hour. The
collector refuses requests above 50; the lead schedules the larger run.

Offline validation: `python3 -m unittest discover -s linux-env/scripts/collect -p
 'test_*.py'`. This includes 1,000 randomized comparisons of legacy pure geometry,
actual shared grasp/jaw execution for both dimension orders/cross settings, the successful fallback-warning probe, late-application and above-rim
rejections, storage thresholds, output confinement and deployment feedback flags.
The CPU-only retroactive checker validates finalized normal batches without
ROS/Isaac/CUDA. It checks ALL episode table rows and tasks; a separate worker
checks two full, deterministic, colour-diverse video episodes per batch and
every video container count. New exports use the same checker on all images.
For example (dataset/pixi Python environment, source batches are read-only):

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
  linux-env/.pixi/envs/default/bin/python -B linux-env/scripts/collect/validate_exports.py \
  --root /mnt/a4500-backup/exploring-vlas/datasets \
  --output /mnt/a4500-backup/exploring-vlas/reports/content-validation/content-validation.json --watch
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 \
  linux-env/.pixi/envs/default/bin/python -B linux-env/scripts/collect/validate_video_exports.py \
  --metadata /mnt/a4500-backup/exploring-vlas/reports/content-validation/content-validation.json \
  --output /mnt/a4500-backup/exploring-vlas/reports/content-validation/video-validation.json --watch
```

The table gate must pass every finalized production batch before training;
sampled video checks may run alongside training and any failure stops training.
The dataset-environment regression is
`python -B linux-env/scripts/collect/export_content_regression.py -v`.

Always validate the completed five-episode smoke before starting the pilot;
check physical feedback, application acknowledgements, containment and dataset
read-back, rather than relying on a scripted completion message.

### Production recovery cohorts

`linux-env/scripts/collect/regrasp_runner.py` owns one assigned instance and
selects complete three-episode RGB cohorts. It invokes the unchanged collector
once per new attempt (`--max-attempts` increases by one), and uses certified
`--resume` only after a healthy accepted predecessor. The instance stays up
within a healthy cohort; its verified initial arm baseline is restored between
collector processes. Each new cohort gets fresh startup and clock certification.

A protected-body velocity screen may reject a start only before any writer
artifact exists. The outer runner replays every completed non-velocity setup
check (candidate selection, measured model/offsets, static clearance, restore epoch/clock/drive
baseline, returned tick contracts and scene checks, then native substeps). It
binds the fault to the captured owned Kit process birth, loaded source, journal
path, epoch range and a fresh before/after observation nonce for that invocation.
An owned PID and birth time plus a held pidfd preserve ownership across startup
process-group changes. A discovery nonce identifies the responder; a second
nonce-bearing observation requires its current kernel identity to remain stable
while that pidfd stays live. Both historical and current groups are retained,
and any refusal records the compared identities. The runner also requires clean
process retirement plus verified owned-down. The terminal
cohort remains untouched as diagnostic data. Its next unused seed starts a fresh
cohort and instance; terminal collector markers are never cleared or relabelled.
Unknown causes, clock/reset/transport/scene failures, any fault after recording
opens, or failed cleanup stop the entire runner.

Every start retains per-body peak velocities and its rejection reason. More than
10% resting-screen rejects stops the runner once at least 20 attempts have run
(exactly 10% is allowed). Only complete RGB cohorts count toward the requested
dataset: 50 cohorts per slot yield 150 selected episodes, 50 of each colour.
`runner.json` also reports physical accepts, unselected valid prefixes, their
collector wall time, and the whole quarantined-cohort extra wall time. Selected
cohorts require native timing/recovery audits, full fresh LeRobot export/load and
all-row/all-decoded-frame equivalence. All cohort files and per-attempt native
observations are sealed, including quarantined prefixes. Every resume reconstructs a bijection between the sealed
receipts, cohort events and the global ledger; it derives selected seed/colour
counts and dataset membership from those verified attempts. Duplicate/orphan
cohorts, conflicting flags, changed seed bases, empty quota claims and modified
files refuse. Any symlink (including dangling links), special entry or linked
ancestor in a cohort tree refuses. Incomplete/pending or mandatory-stop
checkpoints also refuse; version-1 runner checkpoints cannot be resumed.

Before admitting a collector process, the outer runner requires three stable
stage-readiness observations, 0.25 seconds apart, under a 120-second deadline
(plus an already running bounded transport call). Each observation retains the
owned native pidfd/nonce proof, loading status, named prim validity, native-body
path agreement, readable transforms and the exact unchanged collector baseline
query. A failure records path diagnostics and prevents collector launch; it
does not catch or relax errors inside the collector. Start the two assigned
slots at least 60 seconds apart, then retain both owned cleanup proofs.

Production execution is supported only by the qualified `rtx6000` host profile;
other hosts refuse `--go` and capacity admission. These commands prepare plans.
Actual `--go` needs the lead's released slots and launch window, the sealed training pull, and host
ABI/clock/recovery qualification of the exact cut:

```bash
python3 -B linux-env/scripts/collect/regrasp_runner.py --plan \
  --instance 3 --gpu 0 --cohorts 50 --seed-start 600000 --seed-end 604239 \
  --cut "$(git rev-parse HEAD)" \
  --output /mnt/storage3/aaugus11/exploring-vlas/datasets/regrasp-i3

python3 -B linux-env/scripts/collect/regrasp_runner.py --plan \
  --instance 4 --gpu 0 --cohorts 50 --seed-start 650000 --seed-end 654239 \
  --cut "$(git rev-parse HEAD)" \
  --output /mnt/storage3/aaugus11/exploring-vlas/datasets/regrasp-i4
```

Slots 3/4 use domains 93/94, bridges 8783/8785 and HTTP 8784/8786.
The renderer and CUDA are UUID-pinned, nice=10, CPU physics threads=4.
All outputs/caches/runtime stay under the qualified host root. No automatic
backup deletion is implemented. With all cohorts retained, the two slots reserve
20 GiB for recovery raw data/proofs/exports, based on the measured three-episode
smoke with margin. RTX admission before each attempt keeps the **50 GiB floor**,
plus that 20 GiB retention reserve and at least 13 GiB for training (or its larger
declared checkpoint reserve): normally **83 GiB free**, and 20 GiB on root.
The already transferred merged training job is not charged a second time.
The reserve is empirical; a capacity failure stops without starting an attempt.
No source batch, training file, other workspace or archive is cleaned.

The outer runner uses launcher ownership claims and PID/birth/pidfd descendant
tracking. It never adopts a live PID/port, never signals a foreign/reused marker,
keeps cleanup failures latched, and bounds every subprocess stage. Collector
success depends on affirmative receipts, not a shutdown exit code. A recorded
native screen fault remains terminal inside its original collector; only this
outer selection policy permits a new process after the old instance is gone.

Collection notifications run in their own process session on the Mac. Start the
notifier with `python3 linux-env/scripts/collect/notify_collection.py --spawn
--remote-root <rtx-setup-directory> --out <mac-delivery-directory>
--ask-script <session-tools/ask/scripts/ask.py> --recipient so-arm-101-vla`.
The launcher returns only after a fresh remote read and a PID/token/PGID heartbeat.
Keep the returned spawn receipt: a foreground tool session is not a durable
notification owner. The worker catches up from persisted remote stop/final
receipts, records each send before calling ask, and records delivery only after
an accepted outcome. An unknown/failed send remains unresolved and is never
retried automatically. The outbox is bound to its host, run and recipient.
The detached-worker regression delivers a fault after its launcher has exited;
CPU tests do not certify a future Mac outage or app shutdown.

A stopped runner deliberately leaves its completed-run `selected_cohorts`
manifest empty. Previously sealed `cohorts[].selected` entries and
`selected_episodes` are retained, together with their exports and backups.
They are partial results, not a completed production run or a resumable stopped
checkpoint. Verify their seals separately before selecting them for another job.
