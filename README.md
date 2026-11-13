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

The unattended collector uses the same `PickPlaceNode` motion implementation as
control_gui's Quickstart. `grasp_geometry.py` and the calibrated five-DOF solver
in `compute_workspace.py` are independent of ROS and Tk. The collector imports
no Tk, starts no display, and uses neither keyboard events nor spoken announcements.

Collection uses **sim time throughout**: controllers, MoveIt, the expert and
observations follow Isaac `/clock`. Physics advances at 120 Hz, app frames at
30 simulated Hz, and both 640×480 rgb8 ROS2CameraHelper cameras tick at 30 Hz.
The collector joins two images and a JointState with the **same sim stamp**;
each tick supplies one dataset row. This supersedes wall-clock `lerobot-record`
for unattended collection. The collection preset is uncapped; `deploy-clock`
uses the identical simulation/camera configuration with a 30 Hz app ceiling
(target RTF 1.0). Deployment policy replay must use `--fps 30
--action-duration-s 0.03333333333333333`. Measure achieved deployment RTF before
accepting a run; a ceiling does not guarantee the machine can sustain it.

Instance index N selects ROS domain `90+N`, bridge `8777+2N`, HTTP `8778+2N`,
and separate runtime files. `ISAACSIM_GPU` selects the GPU (default 0).
Coordinate assignments before launching on a shared box. DDS uses the
loopback-only CycloneDDS XML; do not combine it with `ROS_LOCALHOST_ONLY=1`.
World timing is selected before scene creation (including the 30 Hz physics
frame-rate clamp). Never change World timing on a live articulation.
Each instance has a portable root; shader caches under `~/.cache/ov` are shared.
Collection writes only beneath `/mnt/a4500-backup/exploring-vlas/` and refuses
to start/continue if `/` has less than 90 GiB free. Never alter the sibling
`a4500-backup-20260828` archive. No source under `isaac/exts/` needs modification.

```bash
# Assigned instance 0: GPU 0, ROS domain 90, bridge 8777, HTTP 8778.
ISAACSIM_GPU=0 bash linux-env/scripts/collect_sim.sh up 0

# Five successful smoke episodes; then validate/export before collecting the pilot.
bash linux-env/scripts/collect_sim.sh run 0 --episodes 5 --seed 310000 \
  --output /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1 \
  --repo-id local/soarm101_headless_smoke \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/smoke-3.1/report

# Pilot only: accepted colours rotate red/green/blue, giving 17/17/16 for 50.
bash linux-env/scripts/collect_sim.sh run 0 --episodes 50 --seed 320000 \
  --output /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1
bash linux-env/scripts/collect_sim.sh export 0 \
  /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1 \
  --repo-id local/soarm101_headless_pilot \
  --report-dir /mnt/a4500-backup/exploring-vlas/datasets/pilot-3.1/report

# Same sim-time stack, paced for policy deployment (collection must be stopped).
bash linux-env/scripts/collect_sim.sh deploy-clock 0
# Shut down only this instance and its control stack.
bash linux-env/scripts/collect_sim.sh down 0
```

The expert follows the collision-validated Quickstart trajectory. At each
camera tick it recomputes a bounded corrective setpoint from measured joints
toward that trajectory; this **clean setpoint** is the recorded action.
Bounded seeded DART disturbances affect only the command sent to the controller,
which is logged separately as `executed_action`. Both use absolute joint positions;
the exported state/action use degrees in deployment's six-joint order, with the
canonical `gripper.pos` feature name. Never run the controller-reference mirror
in this domain: its output contains noisy execution commands and is not the label.
The timeline pauses between motion segments, including MoveIt planning and
attach/detach service calls, so wall-time planning delays do not create long
stationary stretches in simulated demonstrations. Paused app updates are capped
at 10 Hz to leave CPU time for ROS planning. Original command-dispatch clock
stamps are logged separately from the observation tick used by the expert.

`run.json` fixes the baseline scene and held-out eval seeds (900000–900029).
Every attempt has `receipt.json` with its seed, task, offsets, heights, speed,
noise seed, cup parameters, initial poses and physical result. `frames.jsonl`
contains original camera/joint/action sim stamps, wall receipt times, clean actions
and executed commands. Failures retain their receipts but are excluded from the
dataset. A run ends with `summary.json` and `COLLECTION_FINISHED`; check both,
not just the process exit status. `--max-attempts` bounds retries and
`--episode-deadline` bounds each episode in wall time. The collector refuses
requests above 50 episodes; the lead schedules the larger run.

The exporter re-audits every accepted attempt, checks exact video/row counts,
writes through the pinned LeRobotDataset API, replaces its generated timestamp
with the measured relative sim stamp, then loads the result with LeRobotDataset
and reads first/last multimodal samples. It also creates a contact sheet,
three colour sample videos, timing audits and a per-episode manifest.
`--replay-receipt /path/to/attempt-NNNN/receipt.json` regenerates that attempt
from its run baseline and saved parameters (one attempt, one requested success). Do not
accept an episode whose timing audit fails or whose picked block is outside
the matching cup's conservative interior test, or whose observed lift is less
than 25 mm. Report yield, RTF, stationary
frame fraction and episodes per wall-hour alongside the dataset.

Offline checks: `python3 -m unittest discover -s linux-env/scripts/collect -p
 'test_*.py'`. The explicit sim-time trajectory sanity check is
`linux-env/scripts/collect/clock_sanity.py` under system Humble with the assigned
domain/bridge variables; it writes its own slow/fast RTF receipt. LeRobot's
[dataset API](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) documents
schema and video storage; the installed fork is the writer API authority.

Validation on the assigned isolated instance: the five-episode smoke loaded in
LeRobotDataset (4,238 frames, 30 fps; red/green/blue = 2/2/1), with all timing
and clean-expert audits passing. Five of eight attempts were accepted (62.5%).
The shared pure geometry matches the verified GUI mathematics on 1,000 random
cases; 11 offline tests pass. The GUI adapter also boots and serves its widget
inventory and Quickstart services with sim time enabled in the isolated domain.
The collection process itself needs no display.
