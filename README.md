# Exploring-VLAs

A digital twin of the **SO-ARM101** 5-DOF arm in **NVIDIA Isaac Sim 6.1**, driven
by a fine-tuned **SmolVLA** vision-language-action policy. One script brings up
the whole thing on a Linux GPU box:

| Piece | What it is | Runs on |
|---|---|---|
| Isaac Sim + `soarm101-dt` extension | Physics, cameras, legos and cups scene, ROS 2 publishers (`isaac/exts/soarm101-dt`) | the box, GPU 0 |
| omni-kit-mcp bridge | A small TCP server inside Isaac that serves the extension's scene commands as `soarm101.<name>` tools (port 8767), plus a control panel in the Isaac window (`omni-kit-mcp/` submodule) | inside Isaac |
| ROS 2 control stack | MoveIt + ros2_control, plus the opt-in `control_gui` (Tk window) and RViz (`vla_SO-ARM101/`) | the box |
| SmolVLA policy | lerobot async policy server + robot client: async action chunking (`lerobot/` submodule). The RTC flag is passed but currently has no effect; see Known issues | the box, GPU 1 |

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

To switch models without restarting Isaac: `deploy_sim.sh down --policy`, then
`deploy_sim.sh up --model <other>`. The same presets are in `control_gui`'s
Inference tab; if you'd rather start inference from there, bring the stack up
with `up --gui --no-policy` so two policies don't drive the arm at once.

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

## Stopping

`deploy_sim.sh down` stops the policy, then the control stack, then Isaac,
and only the pieces that `up` started (an Isaac or control stack you started
yourself is left alone). `down --policy` stops just the policy.

## Known issues

- **Robot invisible after a cold first launch.** The Isaac log shows
  `usdrt.hydra.fabric_scene_delegate ... cannot find protoPath: /__Prototype_*`.
  Workaround: `isaacsim_launch.sh close`, then `deploy_sim.sh up` again (a warm
  launch renders it). Planned fix: de-instance the robot's visual meshes when
  the scene is built.
- **Sim speed depends on box load.** Real-time factor is ~0.47x on an idle box
  and ~0.31x with other jobs running. The training data was recorded at ~0.32x
  with a ~19.4 Hz wall-clock loop; matching that (rate limit, client fps ~19)
  is planned.
- **RTC (real-time chunking) is not actually applied.** `LEROBOT_RTC_ENABLED=1` logs
  "RTC enabled", but the fork's `policy_server.py` calls `predict_action_chunk(observation)`
  without `prev_chunk_left_over` / `inference_delay`, so RTC returns the unguided output
  (`policies/rtc/modeling_rtc.py`). All runs so far, including April's, were plain async
  chunking. A real RTC fix is planned.
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
