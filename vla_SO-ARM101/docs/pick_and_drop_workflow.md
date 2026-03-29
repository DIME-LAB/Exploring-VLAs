# SO-ARM101 Pick-and-Drop Workflow

End-to-end pick-and-drop for sorting lego blocks into color-matched cups.
Works with Isaac Sim or Gazebo as the simulation backend.

## Architecture

```
                  Isaac Sim                           Gazebo
                  --------                            ------
                  soarm101-dt extension               gazebo.launch.py
                  port 8767                           gz_ros2_control
                       |                                   |
                       v                                   v
              /wrist_camera_rgb_sim              /wrist_camera
              /objects_poses_sim                  /objects_poses_sim
              /drop_poses                        (from aruco localizer)
              /joint_states                      /joint_states
              /ee_pose                           /ee_pose
              /camera_pose                       /camera_pose
                       |                                   |
                       +---------> control_gui <-----------+
                                  (tkinter node)
                                  Pick: grasp tab
                                  Drop: drop LabelFrame
```

Three repos involved:

| Repo | Path | Role |
|------|------|------|
| isaac-sim-mcp | `~/Documents/isaac-sim-mcp` | Isaac Sim extension (sim, cameras, drop pose publishing) |
| aruco_camera_localizer | `~/Desktop/ros2_ws/src/aruco_camera_localizer` | YOLOE object detection + ArUco drop pose detection |
| vla_SO-ARM101 | `~/Projects/Exploring-VLAs/vla_SO-ARM101` | Control GUI, IK, MoveIt, trajectories |

## Prerequisites

```bash
# Build all packages
source /opt/ros/humble/setup.bash

cd ~/Desktop/ros2_ws
colcon build --packages-select aruco_camera_localizer --symlink-install

cd ~/Projects/Exploring-VLAs/vla_SO-ARM101
colcon build --packages-select so_arm101_control --symlink-install
colcon build --packages-select so_arm101_description
```

## Option A: Isaac Sim Backend

### A1. Launch Isaac Sim

```bash
DISPLAY=:0 ~/env_isaaclab/bin/isaacsim \
  --ext-folder ~/Documents/isaac-sim-mcp/exts \
  --enable soarm101-dt
```

Wait ~60s for full startup. Verify the MCP socket responds:

```python
import socket, json, time
s = socket.socket(); s.settimeout(5)
s.connect(('localhost', 8767))
s.sendall(json.dumps({'type': 'list_available_tools', 'params': {}}).encode())
d = b''
t0 = time.time()
while time.time() - t0 < 5:
    c = s.recv(4096)
    if not c: break
    d += c
    try: json.loads(d.decode()); print('RESPONSIVE'); break
    except: continue
s.close()
```

### A2. Quick Start (loads scene)

Via MCP socket or UI "Quick Start" button:

```python
import socket, json
s = socket.socket(); s.settimeout(120)
s.connect(('localhost', 8767))
s.sendall(json.dumps({'type': 'quick_start', 'params': {}}).encode())
d = b''
while True:
    c = s.recv(16384)
    if not c: break
    d += c
    try: json.loads(d.decode()); break
    except: continue
print(json.loads(d.decode()).get('result', {}).get('message', ''))
s.close()
```

Wait ~30s for full scene load (robot, action graphs, objects, simulation start).

### A3. Add Cups + Drop Pose Publisher

```python
# Add cups (via MCP socket)
import socket, json
def send_cmd(tool, params={}, port=8767, timeout=30):
    msg = json.dumps({"type": tool, "params": params})
    s = socket.socket(); s.settimeout(timeout)
    s.connect(("localhost", port))
    s.sendall(msg.encode())
    d = b""
    while True:
        c = s.recv(4096)
        if not c: break
        d += c
        try: json.loads(d.decode()); break
        except: continue
    s.close()
    return json.loads(d.decode())

# Add cups to scene
send_cmd("add_cups")

# Start publishing /drop_poses (ArUco marker poses on cups)
send_cmd("publish_drop_poses")
```

### A4. Move to Grasp Home

```python
send_cmd("execute_python_code", {"code": """
from pxr import UsdPhysics
import omni.usd
stage = omni.usd.get_context().get_stage()
d = UsdPhysics.DriveAPI.Get(
    stage.GetPrimAtPath('/World/SO_ARM101/joints/wrist_flex'), 'angular')
d.GetTargetPositionAttr().Set(90.0)
result = "Moved to grasp home"
"""})
```

Wait 3-5s for physics to settle.

### A5. Verify Topics

```bash
source /opt/ros/humble/setup.bash

# Camera publishing
ros2 topic hz /wrist_camera_rgb_sim --window 5

# Drop poses publishing
ros2 topic echo /drop_poses --once
# Expected: TFMessage with 3 transforms: drop_0, drop_1, drop_2

# Object poses publishing
ros2 topic echo /objects_poses_sim --once
```

### A6. Launch YOLOE Detector

```bash
source ~/Desktop/ros2_ws/install/setup.bash

ros2 run aruco_camera_localizer localize_yoloe \
  --camera-topic /wrist_camera_rgb_sim \
  --yolo-prompts "red block" "blue block" "green block" \
  --headless \
  --yolo-conf 0.25
```

Wait ~20s for model loading. Verify:

```bash
ros2 topic echo /objects_poses --once
```

### A7. Launch Control GUI

```bash
source /opt/ros/humble/setup.bash
source ~/Projects/Exploring-VLAs/vla_SO-ARM101/install/setup.bash

ros2 launch so_arm101_control control.launch.py
```

Wait 5s for GUI initialization.

### A8. Pick-and-Drop Sequence (Isaac Sim)

See "Pick-and-Drop Service Sequence" below.

### A9. Shutdown (Isaac Sim)

```bash
pkill -f "localize_yoloe"
pkill -SIGINT -f "ros2.*launch.*control.launch"
# Isaac Sim: Ctrl+C in terminal or pkill -15 -f "isaacsim"
```

## Option B: Gazebo Backend

### B1. Launch Gazebo + Control Stack

```bash
source /opt/ros/humble/setup.bash
source ~/Projects/Exploring-VLAs/vla_SO-ARM101/install/setup.bash

# Headless (camera still renders)
ros2 launch so_arm101_control gazebo.launch.py headless:=true rviz:=false

# Or with GUI
ros2 launch so_arm101_control gazebo.launch.py rviz:=false
```

Wait ~40s for full startup.

### B2. Move to Grasp Home

```bash
ros2 service call /so_arm101_control_gui/grasp_home std_srvs/srv/Trigger
```

Wait 5-10s for trajectory execution.

### B3. Verify Topics

```bash
ros2 topic hz /wrist_camera --window 5
ros2 topic list | grep -E "wrist|joint_states|ee_pose|camera_pose"
```

### B4. Launch YOLOE Detector

```bash
source ~/Desktop/ros2_ws/install/setup.bash

ros2 run aruco_camera_localizer localize_yoloe \
  --camera-topic /wrist_camera \
  --yolo-prompts "red block" "blue block" "green block" \
  --headless \
  --yolo-conf 0.25
```

### B5. Launch ArUco Drop Detector (for cup markers)

```bash
source ~/Desktop/ros2_ws/install/setup.bash

ros2 run aruco_camera_localizer merged_localization_aruco \
  --camera-topic /wrist_camera \
  --drop \
  --robot so_arm101 \
  --headless
```

This detects ArUco markers on cups (IDs 0,1,2) and publishes `/drop_poses`.

### B6. Pick-and-Drop Sequence (Gazebo)

See "Pick-and-Drop Service Sequence" below.

### B7. Shutdown (Gazebo)

```bash
pkill -f "localize_yoloe"
pkill -f "merged_localization_aruco"
pkill -SIGINT -f "ros2.*launch.*gazebo.launch"
```

## Pick-and-Drop Service Sequence

All services use `std_srvs/srv/Trigger`. Node: `/so_arm101_control_gui`.

```bash
S=/so_arm101_control_gui
T=std_srvs/srv/Trigger
```

### Phase 1: Pick

```bash
# 1. Home position (gripper down, wrist_flex=90 deg)
ros2 service call $S/grasp_home $T

# 2. Load detected objects into pick listbox
ros2 service call $S/grasp_refresh $T

# 3. Select target object (first in list, or set ik_target param)
# To pick a specific object:
#   ros2 param set $S ik_target "red_2x3"
ros2 service call $S/grasp_select $T

# 4. Check reachability
ros2 service call $S/check_grasp_reachable $T

# 5. Open gripper for object (sized to detected bbox)
ros2 service call $S/gripper_open_for_object $T

# 6. Execute pick (approach + descend) — blocks until trajectory completes
ros2 service call $S/grasp_move $T

# 7. Close gripper (grasp object) — blocks until gripper closes
ros2 service call $S/gripper_close_for_object $T

# 8. Return to grasp home (lift object) — blocks until home reached
ros2 service call $S/grasp_home $T
```

### Phase 2: Drop

```bash
# 9. Refresh drop targets from /drop_poses
ros2 service call $S/drop_refresh $T

# 10. Select drop target
# To select a specific target:
#   ros2 param set $S ik_target "drop_0"
ros2 service call $S/drop_select $T

# 11. Point toward cup (rotates shoulder_pan only) — blocks until pan complete
ros2 service call $S/drop_point $T

# 12. Sweep wrist over cup (wrist_flex 90° → 0°) — blocks until sweep complete
ros2 service call $S/drop_sweep $T

# 13. Release object into cup — blocks until gripper opens
ros2 service call $S/drop_release $T

# 14. Return to home — blocks until home reached
ros2 service call $S/grasp_home $T
```

### Full Cycle (copy-paste)

All service calls **block until motion completes** — no sleep timers needed.

```bash
S=/so_arm101_control_gui; T=std_srvs/srv/Trigger

# PICK
ros2 service call $S/grasp_home $T
ros2 service call $S/grasp_refresh $T && sleep 1  # allow topic data
ros2 param set $S ik_target "blue_2x3"
ros2 service call $S/grasp_select $T
ros2 service call $S/gripper_open_for_object $T
ros2 service call $S/grasp_move $T
ros2 service call $S/gripper_close_for_object $T
ros2 service call $S/grasp_home $T

# DROP
ros2 service call $S/drop_refresh $T && sleep 1  # allow topic data
ros2 param set $S ik_target "drop_2"
ros2 service call $S/drop_select $T
ros2 service call $S/drop_point $T
ros2 service call $S/drop_sweep $T
ros2 service call $S/drop_release $T
ros2 service call $S/grasp_home $T
```

> **Note:** Service calls block via `_motion_event` pattern. Each motion command
> uses `_execute_trajectory()` which handles slider sync, feedback suppression,
> and completion signaling. The trigger callback polls `_motion_event` at 0.5s
> intervals and only responds after the full trajectory finishes.

## Drop Configuration

### ArUco Marker to Cup Mapping

| ArUco ID | Drop Frame | Cup Color |
|----------|------------|-----------|
| 0 | drop_0 | red |
| 1 | drop_1 | green |
| 2 | drop_2 | blue |

Dictionary: DICT_4X4_50, marker size: 25mm, placed at 45% cup height.

### Cup Dimensions

| Property | Value |
|----------|-------|
| Diameter | 78mm |
| Height | 96.5mm |
| Marker height | 43.4mm (45% of height) |
| Opening radius | ~39mm |

### Drop Pose Sources

| Source | How | Topic |
|--------|-----|-------|
| Isaac Sim | `publish_drop_poses` MCP tool/button | `/drop_poses` |
| ArUco camera | `merged_localization_aruco --drop --robot so_arm101` | `/drop_poses` |

Both publish `TFMessage` with `child_frame_id = drop_0, drop_1, drop_2`.

### ArUco Config (aruco_config.json)

```json
{
  "active_robot": "so_arm101",
  "robots": {
    "jetank": { "marker_rows": { ... } },
    "so_arm101": {
      "marker_rows": {
        "cups": {
          "marker_ids": [0, 1, 2],
          "position_offset": {"X": 0.0, "Y": 0.083, "Z": -0.039}
        }
      }
    }
  }
}
```

Switch robots: `--robot jetank` or `--robot so_arm101` on CLI,
or set `"active_robot"` in config.

## Verification

### Verify Pick Detection

```bash
# Objects should be visible
ros2 topic echo /objects_poses --once
# Expected: TFMessage with block names and positions
```

### Verify Drop Poses

```bash
ros2 topic echo /drop_poses --once
# Expected: TFMessage with drop_0, drop_1, drop_2
```

### Verify Detection Accuracy (Optional)

```bash
# Isaac Sim ground truth comparison
source ~/Projects/Exploring-VLAs/vla_SO-ARM101/install/setup.bash
ros2 run so_arm101_control verify_detections

# Gazebo ground truth comparison
ros2 run so_arm101_control verify_detections
```

Expected: <5mm average error for both backends.

### Verify Drop Target Reachability

```bash
ros2 param set $S ik_target "drop_0"
ros2 service call $S/drop_select $T
ros2 service call $S/check_grasp_reachable $T
```

## Required Skill

Load the Isaac Sim extension skill at session start for lifecycle scripts:
```
/isaac-sim-extension-dev
```

Lifecycle scripts:
```bash
bash ~/.claude/skills/isaac-sim-extension-dev/scripts/isaacsim_launch.sh {launch|close|kill|restart|status|wait} [ext-id]
bash ~/.claude/skills/isaac-sim-extension-dev/scripts/pick_and_place.sh <pick_object> <drop_target>
```

## Connection Recovery

### Isaac Sim MCP Socket Drop (port 8767)

The MCP socket can die during `delete_cups`/`add_cups` (Blender baking blocks the extension thread) or if `touch extension.py` runs while a socket call is in flight.

**Diagnosis:**
```bash
bash ~/.claude/skills/isaac-sim-extension-dev/scripts/isaacsim_launch.sh status
```

**Recovery:**
1. If process alive but socket dead: open new stage in Isaac Sim GUI (Ctrl+N) to trigger extension re-init, then run `quick_start`
2. If process dead: relaunch with `bash ~/.claude/skills/isaac-sim-extension-dev/scripts/isaacsim_launch.sh launch soarm101-dt`, wait for socket, then `quick_start`

**Prevention:**
- Never `touch extension.py` while making socket calls
- Use 120s timeout for `add_cups` (first-time Blender baking is slow)
- Don't chain rapid socket calls — wait for each response

### arm_controller Spawn Failure (ros2_control)

On ~30% of GUI launches, `arm_controller` fails to configure due to a race with `controller_manager` startup. **The controllers still need to be manually activated.**

**Symptoms:** Services return "executed" but joints don't move. `/joint_states` positions are stuck.

**Diagnosis:**
```bash
ros2 control list_controllers
# If arm_controller shows "unconfigured" instead of "active" → this is the issue
```

**Fix:**
```bash
ros2 service call /controller_manager/configure_controller \
  controller_manager_msgs/srv/ConfigureController "{name: 'arm_controller'}"
ros2 control set_controller_state arm_controller active
ros2 control list_controllers  # verify all 3 show "active"
```

**Verify fix worked:**
```bash
ros2 service call /so_arm101_control_gui/grasp_home std_srvs/srv/Trigger
# Arm should move to grasp home (wrist_flex=90 deg) in Isaac Sim
```

### Service Calls Return Before Motion Completes

Currently `_cmd_*` service calls dispatch trajectories to background threads and return immediately. A scripted sequence of service calls will send conflicting commands because the previous motion hasn't finished.

**Workaround:** Add `sleep` between service calls (3-8s depending on trajectory duration).

**Proper fix (TODO):** Make `_send_arm_goal` block until the FollowJointTrajectory action result arrives.

## Troubleshooting

### No objects detected
- Check camera is publishing: `ros2 topic hz /wrist_camera_rgb_sim`
- Check YOLOE is running: `ros2 topic echo /objects_poses --once`
- Ensure arm is at grasp home (wrist_flex=90 deg) so camera faces down
- If using sim poses only (no YOLOE): check `ros2 topic echo /objects_poses_sim --once`

### Drop poses empty
- Isaac Sim: cups must be added first (`add_cups` via MCP or UI button). `quick_start` now includes this.
- If you deleted cups and re-added: the drop pose action graph was also deleted. Call `publish_drop_poses` again.
- Gazebo: ensure `merged_localization_aruco --drop --robot so_arm101` is running
- Check topic: `ros2 topic echo /drop_poses --once`

### Joints don't move after service call
- Check `ros2 control list_controllers` — all 3 must be "active"
- If `arm_controller` is "unconfigured": see "arm_controller Spawn Failure" above
- Check for multiple GUI instances: `ps aux | grep control_gui | grep -v grep | wc -l` (must be 1)
- Kill duplicates: `ps aux | grep control_gui | grep -v grep | awk '{print $2}' | xargs kill -9`

### Object poses topic empty after scene changes
- If cups were deleted/re-added, the object pose publisher action graph may have been destroyed
- Recreate it: call `setup_pose_publisher` MCP tool or UI button

### "Drop rejected" / not reachable
- Cup is outside workspace (R: 0.054-0.311m, Z: -0.209-0.074m)
- Note: drop operations skip the grasp workspace check (different kinematics)
- If drop_sweep fails with "no IK solution": the drop sweep only moves wrist joints, no IK is computed

### Sweep misses cup
- Offset values in aruco_config.json may need tuning
- Check arm configuration at wrist_flex=0 places gripper above cup
- Adjust sweep duration for smoother motion

### YOLOE model loading slow
- First launch downloads MobileCLIP (~572MB)
- Subsequent launches use cached model
- Wait ~20s after launch for text embedding computation
