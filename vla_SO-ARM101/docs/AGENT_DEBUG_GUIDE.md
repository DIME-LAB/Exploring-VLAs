# Agent Debug Guide — SO-ARM101 Control GUI

How to verify code changes, debug failures, and control the robot through
ROS2 services without touching the GUI manually.

All services use `std_srvs/srv/Trigger` (no arguments, returns `success` + `message`).

Node name: `/so_arm101_control_gui`

## Quick Start

```bash
# Source environment
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash

# Shorthand for all service calls below
alias sc='ros2 service call /so_arm101_control_gui'

# Discover all available commands
sc/list_commands std_srvs/srv/Trigger

# Verify the arm responds
sc/zero_arm std_srvs/srv/Trigger
sc/get_joint_positions std_srvs/srv/Trigger
```

## Build → Launch → Test Loop

```bash
# 1. Build (first time only)
cd ~/ros2_ws
colcon build --symlink-install --packages-select so_arm101_control

# 2. Launch
ros2 launch so_arm101_control control.launch.py

# 3. After code changes in src/:
#    - ANY Python change: hot-reload via Ctrl+R in GUI or service call (reads from src/ directly)
#    - GUI layout changes: Ctrl+Shift+R in GUI
#    - Rebuild only needed for: new files, setup.py/package.xml, entry point changes
```

### When to rebuild vs hot-reload vs restart

| Change | Action | Why |
|--------|--------|-----|
| Method body, constants, imports | **Hot reload** (Ctrl+R or `~/hot_reload` service) | Reloads from `src/` directly, no build needed |
| GUI layout (widgets, tabs, buttons) | **Hot reload GUI** (Ctrl+Shift+R or `~/hot_reload_gui` service) | Rebuilds tkinter widgets with new code |
| New Python files, setup.py, entry points | **`colcon build` + restart** | Package metadata needs updating |
| New `_cmd_*` method | **Hot reload** (method is patched, but service registration needs restart) | Services are registered at `__init__` time |

### Shutdown

```bash
# Clean shutdown (~2s) — always use this
pkill -SIGINT -f "ros2.*launch.*control.launch"

# Never needed: pkill -9. SIGINT propagates cleanly.
# The node uses spin_once() loop that checks running flag,
# so service callbacks exit within 0.5s of signal.
```

## Service Response Behavior

All `_cmd_*` services use `std_srvs/srv/Trigger` and return:
- `success=True` — command completed without warnings/errors
- `success=False` — command failed; `message` contains the error reason

Any `_append_log('...', 'warn')` or `_append_log('...', 'error')` inside a command
automatically sets `success=False` on the service response. Motion commands block
until the trajectory completes before responding.

## Service Reference

### Discovery

| Service | Returns |
|---|---|
| `~/list_commands` | Comma-separated list of all command names |

### State Queries (return data in `message`)

| Service | Returns |
|---|---|
| `~/get_joint_positions` | `name=value` pairs for all 6 joints |
| `~/get_ee_pose` | XYZ + quaternion of end-effector (tcp_link) |
| `~/get_tcp_pose` | XYZ + quaternion from TF lookup |
| `~/get_ik_target` | Current IK tab target (XYZ, RPY, quaternion) |
| `~/get_log` | Full process log (timestamped entries) |

### Arm Control

| Service | Action |
|---|---|
| `~/zero_arm` | Move all joints to 0 |
| `~/randomize_arm` | Random valid joint configuration |
| `~/set_joints` | Send current slider values to controller |
| `~/plan_execute` | MoveIt plan + execute from current to goal |

### IK Tab

| Service | Action |
|---|---|
| `~/ik_randomize` | Randomize IK target + arm |
| `~/ik_set_joints` | Set joints from IK solution |
| `~/ik_plan_execute` | Plan + execute from IK solution |
| `~/set_ik_target` | Set IK target (use `ros2 service call` with empty request, reads from spinboxes) |

### Gripper

| Service | Action |
|---|---|
| `~/gripper_open` | Open gripper to max |
| `~/gripper_close` | Close gripper to min |
| `~/gripper_open_for_object` | Open sized for selected object + clearance |
| `~/gripper_close_for_object` | Close sized for selected object |
| `~/gripper_open_range` | Open by clearance range |
| `~/gripper_close_range` | Close by clearance range |

### Grasp Pipeline

| Service | Action |
|---|---|
| `~/grasp_refresh` | Re-read objects from pose topic |
| `~/grasp_select` | Select first object in listbox |
| `~/check_grasp_reachable` | Check if selected object is within workspace |
| `~/grasp_move` | Full pick sequence: IK solve → approach → descend |
| `~/grasp_home` | Reset grasp state |
| `~/grasp_update_topic` | Switch object pose topic |

### Drop Pipeline

| Service | Action |
|---|---|
| `~/drop_refresh` | Re-read drop targets from /drop_poses topic |
| `~/drop_select` | Select drop target by `ik_target` param (required, no default) |
| `~/drop_point` | Rotate shoulder_pan + set wrist_roll=-90° toward cup |
| `~/drop_sweep` | Geometric IK (45° grip) + MoveIt collision-free path planning |
| `~/drop_release` | Open gripper to release object into cup |

### Clearance Tuning

| Service | Action |
|---|---|
| `~/set_jaw_open_clearance` | Set extra jaw gap on open (mm) |
| `~/set_jaw_close_clearance` | Set extra jaw gap on close (mm) |
| `~/set_tcp_clearance` | Set TCP offset beyond grip width (mm) |

### Scene

| Service | Action |
|---|---|
| `~/toggle_ground_plane` | Toggle MoveIt ground collision plane |

## Pick Sequence (copy-paste)

Complete grasp test for the first reachable object:

```bash
S=/so_arm101_control_gui
T=std_srvs/srv/Trigger

# Reset
ros2 service call $S/zero_arm $T

# Load objects
ros2 service call $S/grasp_refresh $T

# Select and check
ros2 service call $S/grasp_select $T
ros2 service call $S/check_grasp_reachable $T

# Open gripper for object
ros2 service call $S/gripper_open_for_object $T

# Execute grasp (approach + descend)
ros2 service call $S/grasp_move $T

# Close gripper
ros2 service call $S/gripper_close_for_object $T

# Verify
ros2 service call $S/get_ee_pose $T
ros2 service call $S/get_log $T
```

## Drop Sequence (copy-paste)

Drop a held object into a cup (requires /drop_poses publishing):

```bash
S=/so_arm101_control_gui
T=std_srvs/srv/Trigger

# Load drop targets
ros2 service call $S/drop_refresh $T

# Select target (param required)
ros2 param set $S ik_target "drop_1"
ros2 service call $S/drop_select $T

# Point toward cup (pan + wrist_roll=-90°)
ros2 service call $S/drop_point $T

# Drop sweep — geometric IK + MoveIt collision-free path (avoids cup convex hulls)
ros2 service call $S/drop_sweep $T

# Release
ros2 service call $S/drop_release $T

# Return home
ros2 service call $S/grasp_home $T

# Verify
ros2 service call $S/get_log $T
```

For the full pick-and-drop workflow including sim setup, see
`docs/pick_and_drop_workflow.md`.

## Debugging Failures

### "GUI not available"

The GUI hasn't finished initializing yet. Wait 2-3 seconds after launch.

### Command returns success but nothing happens

Check the log — the command dispatches to the tkinter thread. If the GUI is
busy (trajectory executing, MoveIt planning), the 2s timeout may expire before
the command runs.

```bash
ros2 service call $S/get_log std_srvs/srv/Trigger
```

### "No object selected" or empty grasp

```bash
ros2 service call $S/grasp_refresh std_srvs/srv/Trigger
ros2 service call $S/grasp_select std_srvs/srv/Trigger
```

Check that the object pose topic is publishing:
```bash
ros2 topic echo /objects_poses_sim --once
```

### IK solve fails / "not reachable"

The geometric IK workspace for top-down grasps is:
- R: 0.054 — 0.311m (radial distance from pan axis)
- Z: -0.209 — 0.074m (height above base)

Objects outside this annulus can't be grasped with gripper-down constraint.

### Trajectory fails

Check MoveIt is running:
```bash
ros2 service list | grep compute_ik
```

Check controller state:
```bash
ros2 control list_controllers
```

## Adding New Commands

Define a `_cmd_*` method in `control_gui.py`. It auto-registers as a Trigger
service on next launch. Hot reload (Ctrl+R) patches the method body so it
runs with new logic, but the ROS2 service registration only happens at init.

```python
def _cmd_my_new_action(self):
    """Available as ~/my_new_action after restart."""
    # Use _execute_trajectory for any arm motion — handles slider sync,
    # _slider_driven flag, animation, and on_complete callback.
    evt = threading.Event()
    self._motion_event = evt
    self._execute_trajectory(target, duration_s=2.0, on_complete=evt.set)
```

**Motion command pattern:** For collision-aware motions (near cups), use
`_cmd_plan_execute(target=..., on_complete=evt)` which plans via MoveIt OMPL.
For simple motions without obstacles, use `_execute_trajectory()` which does
direct joint interpolation. Never call `_send_arm_goal` directly from
`_cmd_*` methods — that skips slider sync and causes jitter.

For gripper motions, use `_gripper_command(execute=False)` for UI update,
then `_send_gripper_goal(blocking=True)` on a background thread with
`_motion_event` signaling.

No decorators, no registration code. The `_cmd_` prefix is the convention.

## Hot-Reload

Hot reload reads from `src/` directly — no `colcon build` needed.

| Method | Reloads | Use when |
|---|---|---|
| **Ctrl+R** / `~/hot_reload` service | Methods + constants | Changed function logic, IK params, callbacks |
| **Ctrl+Shift+R** / `~/hot_reload_gui` service | Methods + GUI widgets | Changed tab layout, added buttons, spinbox ranges |

Both preserve: ROS2 node, publishers, subscribers, TF, locks, joint state, object data.

**How it works:** `importlib.reload()` redirected to read from `src/` instead of
`build/`. The egg-link from `--symlink-install` points to `build/`, but hot reload
bypasses this by patching `__file__` and `__spec__` before reloading.

**Limitation:** New `_cmd_*` methods are patched onto the instance (callable via
`getattr`), but their corresponding ROS2 service endpoints are only registered
at `__init__`. A restart is needed for new services to appear in `ros2 service list`.
# Button ↔ Service Mapping (auto-generated from dump_services)

> Regenerated from control_gui.py via `ros2 service call /so_arm101_control_gui/dump_services std_srvs/srv/Trigger {}`. Do NOT hand-edit — rerun the service after code changes.

## Buttons

| Tab | Section | Button | Bound Method | Registered Service | Mapping OK |
|-----|---------|--------|--------------|--------------------|------------|
| FK | Arm | Reset Arm | `_cmd_zero_arm` | ~/zero_arm | ✓ |
| FK | Arm | Randomize | `_cmd_randomize_arm` | ~/randomize_arm | ✓ |
| FK | Gripper | Reset Gripper | `_cmd_gripper_zero` | ~/gripper_zero | ✓ |
| FK | Gripper | Open | `_cmd_gripper_open` | ~/gripper_open | ✓ |
| FK | Gripper | Close | `_cmd_gripper_close` | ~/gripper_close | ✓ |
| FK | Actions | Set Joints | `_cmd_set_joints` | ~/set_joints | ✓ |
| FK | Actions | Plan & Execute | `_cmd_plan_execute` | ~/plan_execute | ✓ |
| IK | Arm | Reset Arm | `_cmd_ik_reset` | ~/ik_reset | ✓ |
| IK | Arm | Randomize | `_cmd_ik_randomize` | ~/ik_randomize | ✓ |
| IK | Gripper | Reset Gripper | `_cmd_gripper_zero` | ~/gripper_zero | ✓ |
| IK | Gripper | Open | `_cmd_gripper_open` | ~/gripper_open | ✓ |
| IK | Gripper | Close | `_cmd_gripper_close` | ~/gripper_close | ✓ |
| IK | Actions | Set Joints | `_ik_btn_set_joints` | (wrapper) | WRAPPER |
| IK | Actions | Plan & Execute | `_ik_btn_plan_execute` | (wrapper) | WRAPPER |
| Grasp | Topic | Update Topic | `_cmd_grasp_update_topic` | ~/grasp_update_topic | ✓ |
| Grasp | Topic | Refresh | `_cmd_grasp_refresh` | ~/grasp_refresh | ✓ |
| Grasp | Arm | Home | `_cmd_grasp_home` | ~/grasp_home | ✓ |
| Grasp | Arm | Move to Grab | `_cmd_grasp_move` | ~/grasp_move | ✓ |
| Grasp | Gripper | Grasp Open | `_cmd_gripper_open_for_object` | ~/gripper_open_for_object | ✓ |
| Grasp | Gripper | Grasp Close | `_cmd_gripper_close_for_object` | ~/gripper_close_for_object | ✓ |
| Grasp | Gripper Range | Open | `_cmd_gripper_open_range` | ~/gripper_open_range | ✓ |
| Grasp | Gripper Range | Close | `_cmd_gripper_close_range` | ~/gripper_close_range | ✓ |
| Grasp | Drop Source | Update Drop Topic | `_drop_btn_update_topic` | (wrapper) | WRAPPER |
| Grasp | Drop Source | Refresh | `_cmd_drop_refresh` | ~/drop_refresh | ✓ |
| Grasp | Drop | Point to Drop | `_cmd_drop_point` | ~/drop_point | ✓ |
| Grasp | Drop | Sweep to Drop | `_cmd_drop_sweep` | ~/drop_sweep | ✓ |
| Grasp | Drop | Release | `_cmd_drop_release` | ~/drop_release | ✓ |
| RViz | Cups | Apply | `_cmd_apply_collision_padding` | ~/apply_collision_padding | ✓ |
| Logs | Process | Clear | `_log_btn_clear` | (wrapper) | WRAPPER |
| Logs | Process | Copy | `_log_btn_copy` | (wrapper) | WRAPPER |
| Logs | Errors | Clear | `_log_btn_clear` | (wrapper) | WRAPPER |
| Logs | Errors | Copy | `_log_btn_copy` | (wrapper) | WRAPPER |

## Agent-only services (no button binding)

| Service | Method |
|---------|--------|
| `~/check_grasp_reachable` | `_cmd_check_grasp_reachable` |
| `~/drop_select` | `_cmd_drop_select` |
| `~/error` | `_cmd_error` |
| `~/grasp_select` | `_cmd_grasp_select` |
| `~/hot_reload` | `_cmd_hot_reload` |
| `~/hot_reload_gui` | `_cmd_hot_reload_gui` |
| `~/ik_plan_execute` | `_cmd_ik_plan_execute` |
| `~/ik_set_joints` | `_cmd_ik_set_joints` |
| `~/set_jaw_close_clearance` | `_cmd_set_jaw_close_clearance` |
| `~/set_jaw_open_clearance` | `_cmd_set_jaw_open_clearance` |
| `~/set_tcp_clearance` | `_cmd_set_tcp_clearance` |
| `~/toggle_ground_plane` | `_cmd_toggle_ground_plane` |

---

# Widgets (auto-generated from list_widgets)

> Regenerated from control_gui.py via `ros2 service call /so_arm101_control_gui/list_widgets std_srvs/srv/Trigger {}`. Do NOT hand-edit.

## Widgets

| Tab | Section | Label | Type | Current Value | Writable |
|-----|---------|-------|------|---------------|----------|
| - | Scene | `Ground Plane` | Checkbutton | true | yes |
| - | Scene | `Ground Z` | Spinbox | 0.0 | yes |
| FK | Actions | `Speed` | Spinbox | 0.5 | yes |
| FK | Joint Sliders | `elbow_flex` | Scale | 0.0 | yes |
| FK | Joint Sliders | `gripper_joint` | Scale | 0.0 | yes |
| FK | Joint Sliders | `shoulder_lift@FK` | Scale | 0.0 | yes |
| FK | Joint Sliders | `shoulder_pan` | Scale | 0.0 | yes |
| FK | Joint Sliders | `wrist_flex` | Scale | 0.0 | yes |
| FK | Joint Sliders | `wrist_roll` | Scale | 0.0 | yes |
| Grasp | Arm | `Approach height (m)` | Spinbox | 0.05 | yes |
| Grasp | Arm | `Arm Duration (s)` | Spinbox | 2.5 | yes |
| Grasp | Arm | `Cross-axis grasp` | Checkbutton | false | yes |
| Grasp | Arm | `Object Z (m)` | Spinbox | 0.0 | yes |
| Grasp | Detected Objects | `Detected Objects` | Listbox |  | yes |
| Grasp | Drop | `Sweep Duration (s)` | Spinbox | 2.5 | yes |
| Grasp | Drop Source | `Drop Topic` | Entry | /drop_poses | yes |
| Grasp | Drop Targets | `Drop Targets` | Listbox |  | yes |
| Grasp | Gripper | `Close clearance (mm)` | Spinbox | 0.0 | yes |
| Grasp | Gripper | `Grip Close` | Spinbox | -10.0 | yes |
| Grasp | Gripper | `Grip Duration (s)` | Spinbox | 3.0 | yes |
| Grasp | Gripper | `Grip Open` | Spinbox | 100.0 | yes |
| Grasp | Gripper | `Open clearance (mm)` | Spinbox | 5.0 | yes |
| Grasp | Gripper | `TCP clearance (mm)` | Spinbox | 1.0 | yes |
| Grasp | Topic | `BBox Topic` | Entry | /objects_bbox_sim | yes |
| Grasp | Topic | `Grasp Topic` | Entry | /objects_poses_sim | yes |
| Grasp | Topic | `TCP offset` | Checkbutton | true | yes |
| IK | Joint Sliders | `gripper_joint@IK` | Scale | 0.0 | yes |
| IK | Orientation | `Roll` | Spinbox | 90.0 | yes |
| IK | Orientation | `Yaw` | Spinbox | 90.0 | yes |
| IK | Orientation | `shoulder_lift@IK` | Spinbox | 87.2 | yes |
| IK | Position | `X` | Spinbox | 0.1 | yes |
| IK | Position | `Y` | Spinbox | 0.05 | yes |
| IK | Position | `Z` | Spinbox | 0.18 | yes |
| RViz | Cups | `Collision padding %` | Spinbox | 10 | yes |
| RViz | Cups | `Visual (colored cups)` | Checkbutton | true | yes |
| RViz | Planning | `Drop grip angle (deg)` | Spinbox | 45 | yes |
| RViz | Planning | `Planning attempts` | Spinbox | 50 | yes |

## Agent-only widgets (no tab/section)

| Label | Type | Current Value | Writable |
|-------|------|---------------|----------|
| `error_log` | Text |  | no |
| `log_tab` | Notebook | Process Log | yes |
| `process_log` | Text | [23:44:01] Ground plane add… | no |
| `tab` | Notebook | IK | yes |
