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
physical state and pose snapshot at tick t, it computes a clean corrective
setpoint, independently adds seeded DART noise to the execution array, and writes
the execution targets into the existing PhysX drives while paused. It reads
those targets back, logs the actual native simulation-clock application stamp,
and advances exactly one frame. A ROS command/acknowledgement protocol applies
each command once. Physical snapshots and motion workers carry immutable
cancellation epochs. One ownership lock covers computation, dispatch, ACK,
recording and completion delivery. Each completion event owns its verdict;
waiting workers revalidate their original epoch after waking. Attach/detach,
scene updates, carried-object bookkeeping and status writes validate and execute
under that same ownership lock, so cancelled workers cannot affect a later episode.
Cancellation/reset waits up to 15 wall seconds for ownership to retire. A stalled
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
the checked-in SRDF. Final verification requires exact expected world IDs/poses,
no attached objects, the correct ground, measured robot state and canonical ACM.
ACM comparison checks off-diagonal pairs: missing names/pairs mean false and
self-pairs are irrelevant. All robot/world permissions remain strict, including
rejection of defaults. The CPU service fake models MoveIt 2.5.10's
[constructor](https://github.com/moveit/moveit2/blob/2.5.10/moveit_core/collision_detection/src/collision_matrix.cpp#L66-L96)
and [serializer](https://github.com/moveit/moveit2/blob/2.5.10/moveit_core/collision_detection/src/collision_matrix.cpp#L332-L361).
Final verification checks each rebuilt primitive's type/dimensions and each
mesh's vertices/triangle identity against the shared shape builders, with
1e-7 m numeric tolerance and harmless mesh indexing/order changes accepted.
Both cleanup and final rebuild require empty occupancy. The first startup reset
writes `first-reset-scene.json` in the run directory, including the native ACM,
world/attached IDs, occupancy size and geometry checks. Accepted receipts retain
the same cleanup and verification results. Include this file in the smoke report;
CPU serialization tests do not establish the live MoveIt round trip.
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

Success uses a measured height-dependent interior profile from the full cup
STL. `cup_profile.json` contains a concave envelope verified against every active
triangle supporting plane, with 0.1 mm subtracted at the calibration nodes and
an additional 0.7 mm inward geometric allowance. Resolved tensor values are
block contact 1.3625 mm/rest 0 and cup-shape contact 0.1 mm/rest 0. Positive contact
offsets trigger predictive contacts; they do not define permitted wall overlap.
The geometric allowance is independent of their sum. All eight oriented block corners must fit their
height's radius. Concavity makes the admitted volume convex, so fitting the
corners also bounds the box between them. The 0.6 mm lower gate, 94 mm ceiling,
15 settled ticks and 25 mm lift requirement remain. Receipts log each corner's
height, radial bound and clearance; run metadata includes the profile and source
hashes. A changed cup STL refuses collection until recalibrated.
See [cup geometry and calibration](docs/notes/collection-success-geometry.md).

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
and must match on resume. See the [complete wait sweep](docs/notes/collection-wait-clocks.md).

`deploy-clock` restores the ROS follower, keeps physical feedback publishing,
and caps the identical simulation/camera profile at 30 app Hz (target RTF 1.0).
`deploy` also starts the policy launcher with **fps=30**, action duration=1/30 s
and `--robot.joint_states_topic=/joint_states_measured`. The existing LeRobot
robot config already supports this topic option; real hardware keeps its measured
`/joint_states` default. A cap does not guarantee throughput: measure actual RTF
before accepting deployment. Legacy eval defaults remain separate from this preset.

Instance N selects domain `90+N`, bridge `8777+2N`, HTTP `8778+2N`, policy RPC
`18090+N` and private runtime files. `ISAACSIM_GPU` selects renderer, physics
CUDA and CUDA consumers (default 0); Isaac leaves CUDA_VISIBLE_DEVICES unset.
Use only assigned resources on shared boxes. DDS uses loopback-only CycloneDDS;
do not also set ROS_LOCALHOST_ONLY. World timing is set before scene creation,
including the 30 Hz frame-rate clamp. Never change World timing on a live scene.

All run/export/runtime paths resolve beneath the canonical
`/mnt/a4500-backup/exploring-vlas/`; environment overrides stay within it.
ROS logs, temp, HF/OptiX caches, portable roots and per-instance grasp traces
are directed there. The indexed launcher overrides inherited
`MCP_CLIENT_OUTPUT_DIR` with its confined instance resources directory and validates
the resolved path before creating it. Direct Python entrypoints apply confinement before imports that
create trace/log/cache files. Shared shaders under `~/.cache/ov` remain shared. Require
50 GiB on the output filesystem before each attempt; below that, wait every
60 s until 52 GiB is available. Root has a **20 GiB last-resort stop**, including
launcher startup: the earlier 90 GiB rule protected root writes, while collection
bulk data lives on another filesystem. Never alter the sibling backup archive.
Receipts measure growth of writable root files in this instance's process trees,
identifying shared caches. This is space growth, not in-place rewrite traffic.

```bash
# Use the box/domain/GPU assigned by the lead. Instance 0 here is GPU 0/domain 90.
ISAACSIM_GPU=0 bash linux-env/scripts/collect_sim.sh up 0

bash linux-env/scripts/collect_sim.sh clock-sanity 0 \
  --output /mnt/a4500-backup/exploring-vlas/runtime/instance-0/clock-sanity.json

bash linux-env/scripts/collect_sim.sh run 0 --episodes 5 --seed 310000 --diagnostic \
  --output /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-physical-v5
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-physical-v5 \
  --repo-id local/soarm101_headless_smoke \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1-physical-v5/report

# Only after the corrected five-episode smoke passes; accepted colours rotate 17/17/16.
bash linux-env/scripts/collect_sim.sh run 0 --episodes 50 --seed 320000 \
  --output /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v2
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v2 \
  --repo-id local/soarm101_headless_pilot \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1-v2/report

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
The concave height-dependent radial profile and 94 mm ceiling bound the block
inside the authored shell (rim 96.52 mm). The lower gate remains 0.6 mm, an
explicit geometric allowance below the authored 1.2 mm floor. Resolved rest
offsets are zero; a negative authored/default UI parameter is not used to infer
physical overlap. Floor-supported placements pass; below-floor, side-protruding
and above-rim placements fail. [PhysX contact/rest semantics](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/_api_build/class_px_shape.html)
separate predictive contact distance from resting geometry. Confirm the cooked
USD collision floor and actual rest height during the smoke;
the offline asset fixture alone does not establish that live measurement.

State/action use absolute degrees in deployment's six-joint order, including
`gripper.pos`; raw audit rows retain radians. The action is the clean corrective
expert command. Seeded disturbances affect only `executed_action`; actual PhysX
read-back targets are logged separately. Noise uses std 0.002–0.006 rad,
probability 0.30, clip ±0.012 rad, during the first 80% of arm trajectories;
the final 20% is clean recovery. Diversity also covers object poses, cup line/arc
layouts and colour order, approach offsets/heights and speed.

`summary.json` must have complete=true and COLLECTION_FINISHED; exit code alone
is insufficient. In-progress checkpoints are explicitly marked. Episode/step
wall deadlines fail bad attempts. Failed receipts remain; failed videos are
excluded. The exporter refuses partial/superseded runs, re-audits original ticks,
checks exact row/video counts, writes measured relative sim timestamps through
the pinned LeRobotDataset API, reloads the dataset, and creates a contact sheet,
three colour sample videos and parameter/timing reports. Report accepted yield,
RTF, idle/stationary fractions, storage growth and episodes per wall-hour. The
collector refuses requests above 50; the lead schedules the larger run.

Offline validation: `python3 -m unittest discover -s linux-env/scripts/collect -p
 'test_*.py'`. This includes 1,000 randomized comparisons with frozen verified-GUI
geometry, the successful fallback-warning probe, late-application and above-rim
rejections, storage thresholds, output confinement and deployment feedback flags.
Always validate the completed five-episode smoke before starting the pilot;
check physical feedback, application acknowledgements, containment and dataset
read-back, rather than relying on a scripted completion message.
