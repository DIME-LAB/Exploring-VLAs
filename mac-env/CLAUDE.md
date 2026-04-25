# mac-env — agent context

This dir is the **macOS / pixi build environment** + **session-management scripts** for the SO-ARM101 stack.

> **Inherits**: see `../CLAUDE.md` for repo topology, build paths, macOS gotchas.

## Why `mac-env/` exists

RoboStack ships ROS2 Jazzy as conda packages. Conda-shipped binaries embed `-isystem` paths in their compiler shebangs, which **break on paths with spaces** (this project lives under `~/Documents/Projects/untitled folder/...`). The bootstrap copies `pixi.toml` + `cyclonedds.xml` to **spaceless `/tmp/mac-env/`** and runs `pixi install` there — and similarly creates the colcon workspace at `/tmp/soarm-ws/` with symlinks back to source. Don't try to build in-tree, it'll fail.

## Script lifecycle (who owns what)

The shell scripts split cleanly into **producers** (start/stop the sim or real producers of `/joint_states`, `/wrist_camera`, etc.) and **consumers** (record from those topics). Producers outlive consumers on purpose — typical workflow is one `stack_start` then many `record`s.

| Script | Lifecycle role | Detached? |
| --- | --- | --- |
| `bootstrap.sh` | One-time setup: pixi env at `/tmp/mac-env`, colcon ws at `/tmp/soarm-ws`, clone aruco_camera_localizer, build everything | Foreground |
| `stack_start.sh` | Start sim stack (`gz sim` + ros2_control + MoveIt + control_gui + parameter_bridge). Modes: `headless` / `gz` / `rviz` / `all` | **Detached** (writes pidfile) |
| `stack_stop.sh` | SIGINT the stack (propagates to children) | Foreground |
| `stack_status.sh` | Print running stack-related processes | Foreground |
| `stack_restart.sh` | stop + start | Foreground |
| `record.sh` | Subscriber-only — wraps `lerobot-record` with the Mac env vars + 2-camera default. **Aliased symlink: `record_sim.sh → record.sh`** for back-compat | Foreground |
| `record_one_shot.sh` | Owns the whole record session: starts producers (camera + drive script or jointstatereader), runs `record.sh`, SIGINTs every child on exit. `--real` switches to hardware-driven topology | Foreground |
| `drive_joint_commands.py` | Sim-side teleop stand-in — publishes a slow shoulder_pan sine sweep on `/joint_commands` at 30 Hz. Use during sim records when there's no leader arm | Foreground |
| `drive_base_yaw_sweep.py` | Phase 5 motion test — automated yaw sweep + gripper toggle | Foreground |
| `lerobot-record-mode.sh` | Internal `--mode sim|real` shim — prepends the right `--robot.type` + `--teleop.type` flags and execs `lerobot-record` | Internal |
| `verify_parity.py` | Compare a recorded dataset's schema against a target HF dataset; exit 0 on match | Foreground |
| `verify_sim_topics.sh` | Print `ros2 topic list/hz` for the topics a record run cares about | Foreground |
| `smoke_publisher.py`, `smoke_l1_camera.py`, `smoke_l2_dataset.py` | Phase 1/2 plumbing smoke tests | Foreground |
| `_lib.sh` | Sourced by stack_*.sh — exports the canonical paths (`MAC_ENV_DIR=/tmp/mac-env`, `SOARM_WS=/tmp/soarm-ws`) and runtime env (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `KMP_DUPLICATE_LIB_OK=TRUE`) | Library |

## Two record paths (pick one)

```
record.sh                        record_one_shot.sh
─────────                        ──────────────────
producers up?                    starts producers itself
  stack_start.sh                   (sim stack if needed,
  drive script                      camera_publisher,
  ↓                                 drive script or jsr)
record.sh ...                    runs record.sh
  ↓                                 ↓
record exits                     SIGINTs every child
producers KEEP RUNNING           (cleanly tears down)
(you bash stack_stop.sh later)   (sim stack stays up if WE started it)
```

`record.sh` is the right choice for tight iteration (record many datasets without re-bringup). `record_one_shot.sh` is the right choice for "record one and walk away" — the cleanup is a real ergonomic win.

## Back-compat: `record_sim.sh` → `record.sh`

`record_sim.sh` is a **symlink** to `record.sh` (renamed in Phase 6-03 because the wrapper now serves both sim and real modes). Don't break the symlink — every existing doc and PR that references `record_sim.sh` keeps working through it.

## Conventions

- **All scripts assume** `/tmp/mac-env/pixi.toml` exists and `/tmp/soarm-ws/install/setup.bash` is sourceable. Re-bootstrap if either is missing.
- **Don't add `pkill -9` patterns** like `"ros2|gz"` — they match terminal processes and IDE windows. Always `pkill -SIGINT -f "ros2.*launch"` or kill specific exact-named procs (per project CLAUDE.md kill rule).
- **All `pixi run` invocations need `--manifest-path /tmp/mac-env/pixi.toml`** because the env doesn't live in-tree (spaced-path constraint).
- **New scripts source `_lib.sh`** for the canonical paths instead of hardcoding `/tmp/...`.
