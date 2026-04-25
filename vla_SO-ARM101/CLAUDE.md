# vla_SO-ARM101 — SO-ARM101 ROS2 Control Stack

ROS2 Humble packages for the SO-ARM101 5-DOF arm: MoveIt2 motion
planning, tkinter control GUI, geometric IK, drop motion pipeline.
README has full project context — this file captures Claude-specific
working knowledge that's not in the README.

## Cross-repo

Isaac Sim digital twin lives at `~/Documents/isaac-sim-mcp/` (branch
`so-arm101`). MCP socket on `localhost:8767`. The pick-and-drop
pipeline spans both repos — this one plans + executes motion;
isaac-sim-mcp provides physics, the cup/lego scene, and the
`/drop_poses` + `/objects_poses_sim` topics. Bring-up sequence and
MCP tool inventory live in `~/Documents/isaac-sim-mcp/CLAUDE.md`.

## Build & launch

```bash
# Build only this stack (symlink-install picks up Python edits without rebuild)
colcon build --packages-select so_arm101_control --symlink-install

# Launch full stack (mock hardware + RViz + MoveIt + control GUI + MTC)
ros2 launch so_arm101_control control.launch.py rviz:=true mtc:=true

# Restart only this stack (leaves Isaac Sim alone — graceful pkill -SIGINT inside)
~/Documents/isaac-sim-mcp/scripts/restart-control-stack.sh
```

The restart script lives in the OTHER repo because it predates this
one. It's been the canonical restart path for both repos' work.

## Tiered deterministic motion planner

Every arm motion routes through `_joint_space_collision_free_execute`
(`src/so_arm101_control/so_arm101_control/control_gui.py:5817`):

1. **Tier 1** — `_plan_linear_joint_path` (line 5923) builds an
   N-waypoint linear interp, validates each via
   `_check_state_valid_with_contacts`, and dispatches the SAME N
   waypoints to `_execute_full_trajectory`. The validated path IS
   the executed path — no spline divergence between knots.
2. **Tier 2** — `_plan_retract_pan_settle` (line 5960) decomposes
   into 3 linear sub-segments: retract to `NEUTRAL_NON_PAN`, pan
   across, settle to target. Each sub-segment passes through
   tier-1; concatenated into one trajectory if all three clear.
3. **OMPL fallback** — `_ompl_plan_validate_execute` (line 6133).
   Opt-in per caller via `allow_ompl_fallback=True` (default).
   `_cmd_grasp_home` (line 5755) passes `False` — must use
   deterministic tiers, surface failure loudly instead of rolling
   RNG dice.

Per-motion `tracer.event('planner_used', which='linear|retract_pan_settle|ompl_fallback|none')`
records which planner ran. Read in /tmp/arm_traj/*.json or via
`/so_arm101_control_gui/get_log` service.

## Drop motion specifics

- **Attach offset captured at grasp time**: `_attach_lego_to_gripper`
  (line 5168) computes block-in-tcp_link pose and stores it on
  `self._attached_lego_tcp_offset`. `_cmd_drop_sweep` (line 6793)
  uses MEASURED `|ax|` for the gap→tcp shift, NOT theoretical
  `half_gap`. Off-center grasps shift the block 1-2 mm vs theoretical
  jaw center; planner compensates.
- **Pan lock on drop_sweep**: `_plan_collision_free_execute`
  (line 2176) accepts `lock_pan=<rad>` kwarg. drop_sweep passes
  current shoulder_pan so geometric IK on the shifted tcp_target
  doesn't introduce ~1° base yaw vs drop_point's pan. Mirrors
  `find_reachable_grasp_yaw`'s "single yaw across stages" pattern.
- **Hover above cup rim default 50 mm**. Block clears cup wall
  during the wrist_flex 90° → 55° sweep arc. User-tunable via
  Grasp tab → Drop section spinbox.
- **Cup padding default 5%** (`_CUP_COLLISION_PADDING = 1.05` at
  line 172). MoveIt planning scene only — Isaac Sim physics sees
  the real cup geometry. Absorbs controller tracking-lag overshoot
  during fast pan motions (post-drop grasp_home is the worst case
  at ~36°/s peak pan rate).

## Test harness (`scripts/`)

```bash
# Single QS cycle for one named lego
scripts/test_qs_cycle.sh red_2x3 [/tmp/log_path]

# Reset Isaac Sim scene + control_gui state to a clean baseline
scripts/sim_reset.sh
# Internally: qs_restart → detach_lego → MCP update_cups
#           → MCP randomize_object_poses → qs_refresh_all → grasp_home

# Sequentially pick-and-place every available lego
scripts/test_pick_all.sh /tmp/run_$(date +%Y%m%dT%H%M%S)
# Writes per-lego logs + summary.csv with PASS/FAIL + planner counts
```

`test_qs_cycle.sh` polls `/so_arm101_control_gui/get_log` for terminal
sentinels (`pick-and-drop cycle complete | Quickstart halted at |
Quickstart aborted`) and trims the captured log to per-cycle scope at
exit, so downstream consumers see only this-cycle counts.

## Key files

| Method / constant | File:Line | Role |
|---|---|---|
| `_joint_space_collision_free_execute` | `control_gui.py:5817` | Tiered planner entry point |
| `_plan_linear_joint_path` | `control_gui.py:5923` | Tier-1 linear interp + validate |
| `_plan_retract_pan_settle` | `control_gui.py:5960` | Tier-2 decomposition |
| `_ompl_plan_validate_execute` | `control_gui.py:6133` | Tier-3 OMPL with Mode A/B classification |
| `_plan_collision_free_execute` | `control_gui.py:2176` | IK + collision check + dispatch (drop_sweep, IK targets) |
| `_cmd_grasp_home` | `control_gui.py:5755` | Pure deterministic; `allow_ompl_fallback=False` |
| `_cmd_drop_sweep` | `control_gui.py:6793` | IK-planned with measured offset + pan lock |
| `_attach_lego_to_gripper` | `control_gui.py:5168` | Captures `_attached_lego_tcp_offset` |
| `_QS_SEQUENCE` | `control_gui.py:3424` | The 9-step pick-and-drop player flow |
| `_CUP_COLLISION_PADDING` | `control_gui.py:172` | Module-level cup pad multiplier |

## Gotchas

- **Tier-1 spline-divergence trap**: an earlier tier-1 implementation
  validated N waypoints but dispatched only `(start, end, duration)`
  — controller's spline took different paths. Current tier-1
  dispatches the FULL N-waypoint trajectory. Don't "optimize" by
  collapsing back to (start, end), and don't reuse `_execute_trajectory`
  (line 7167) for tier-1 — it's the (start, end, duration) path.
  Use `_execute_full_trajectory` (line 6249).
- **Padding asymmetry is intentional**: the MoveIt planning scene
  has padded cups, Isaac Sim physics doesn't. Don't compare
  `/check_state_validity` results to omni.physx contact telemetry
  as if they should agree — they intentionally don't.
- **`home_velocity_scale` is OMPL-only**: the param overrides
  `velocity_scale_var`, but `velocity_scale_var` is consumed only
  by `_ompl_plan_sync` (line 6228). Tier-1 / tier-2 use `duration_s`
  directly. So `home_velocity_scale` is dead code for the
  deterministic planner path. To slow grasp_home, edit `duration_s`
  on the `_joint_space_collision_free_execute` call inside
  `_cmd_grasp_home`.
- **MCP cup tools live across the repo boundary**: `update_cups`,
  `randomize_object_poses`, `sort_into_cups` are MCP tools in the
  isaac-sim-mcp repo, called via socket on localhost:8767. See
  `scripts/sim_reset.sh` for the call pattern.
- **Convention-based debug services**: every `_cmd_*` method on the
  control_gui node auto-registers as a `/so_arm101_control_gui/<name>`
  Trigger service (the leading `_cmd_` is stripped). GUI button
  click and service call hit the SAME code via `_make_trigger_callback`
  (line 864) which schedules the call on the Tk thread. Both paths
  clear `_motion_event` / `_last_motion_status` / `_cmd_error`
  before dispatch. So the script harness and GUI buttons produce
  identical behavior.

## Tracking-lag context

`docs/notes/2026-04-24-forum-findings-tracking-lag.md` (in
isaac-sim-mcp repo) documents the SO-ARM101 controller tracking-lag
investigation that motivated the deterministic planner + 5% padding
combination. wrist_flex tracks 10-13° behind commanded position at
high velocities; shoulder_pan accumulates 22° at peak ref_vel. The
planner produces conservative paths, padding absorbs residual lag,
and tier-2 decomposition handles the "post-drop grasp_home swings
near another cup" scenario.
