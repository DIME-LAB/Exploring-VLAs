# SO-ARM101 Grasp Pipeline

Technical reference for the top-down grasp pipeline used by the SO-ARM101 control GUI.

## References

- [Maegan Tucker — ECE4560 Assignment 7: SO-101 IK](https://maegantucker.com/ECE4560/assignment7-so101/) — IK frame diagram and geometric decomposition
- [TheRobotStudio SO-ARM100 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf) — kinematic chain reference
- [MuammerBay SO-ARM_ROS2_URDF](https://github.com/MuammerBay/SO-ARM_ROS2_URDF) — joint naming convention

## Overview

The grasp pipeline moves the arm to pick up a detected object using a two-stage
IK approach (approach then descend), with a single-moving-jaw gripper whose TCP
sits at the fixed jaw tip.

**Both stages are pre-validated before the arm moves.** If either the approach
or final pose fails geometric IK or collision checking, the entire grasp is
rejected and the arm stays in place.

### Kinematic Chain

```
... -> wrist_roll -> gripper (body) -> tcp_link   (FIXED joint)
                                    -> jaw         (REVOLUTE - gripper_joint)
```

TCP is at the fixed jaw tip. The moving jaw swings open/closed via
`gripper_joint`. Opening the jaw does **not** move the TCP.

## Geometric IK

The top-down grasp uses an analytical geometric IK solver instead of MoveIt KDL.
There is **no MoveIt fallback** for grasp moves — if geometric IK fails, the
position is unreachable for a top-down grasp.

### Solver Steps

```
1. theta1 (pan)       = atan2(-y, x - X_PAN)
2. theta5 (wrist_roll)= theta1 + grasp_yaw - WRIST_ROLL_OFFSET
3. Back-compute wrist_flex pivot from TCP target (roll-adjusted offset)
4. theta2, theta3     = 2-link law of cosines (lift, elbow)
5. theta4 (wrist_flex)= pi/2 - theta2 - theta3   [gripper-down constraint]
6. FK refinement step to compensate for cross-plane coupling (~1-8mm error -> <0.5mm)
7. Joint limit check
8. Collision check via MoveIt GetStateValidity
9. Tries elbow-up first, then elbow-down
```

### Gripper-Down Constraint

The constraint `theta2 + theta3 + theta4 = pi/2` (90 degrees) ensures the gripper
always points straight down. This eliminates one degree of freedom — `theta4` is
fully determined by `theta2` and `theta3`. The constants `WF_TCP_DR` and
`WF_TCP_DH` in `compute_workspace.py` are derived assuming exactly pi/2.

### IK Constants

```
X_PAN       = 0.0388 m     Pan joint X offset from base origin
LIFT_R      = 0.0304 m     Shoulder-lift pivot radial distance
LIFT_H      = 0.1166 m     Shoulder-lift pivot height
L_UPPER     = 0.1160 m     Upper arm length (lift to elbow)
L_LOWER     = 0.1350 m     Lower arm length (elbow to wrist)
WF_TCP_DR   = -0.0079 m    Wrist-flex to TCP radial offset (gripper-down)
WF_TCP_DH   = -0.15923 m   Wrist-flex to TCP height offset (gripper-down)
WRIST_ROLL_OFFSET = pi/2 - 0.0487 rad
```

## Grasp Workspace

The top-down grasp workspace is **much smaller** than the general workspace
because the gripper-down constraint limits which joint configurations are valid.

### How It Is Computed

The general workspace is computed via Monte Carlo FK sampling (random joint
angles, compute TCP position). The grasp workspace is computed differently —
it sweeps a grid of `(r, z, yaw)` target positions through `geometric_ik()`
and records which positions actually produce valid solutions. This accounts for
the coupling between `theta1`, `theta5`, and the target position that FK
sampling misses.

```
General workspace:  random joints -> FK -> TCP positions      (any orientation)
Grasp workspace:    grid of targets -> geometric_ik() -> pass/fail  (top-down only)
```

### Workspace Bounds

```
+-------------------+-----------------------+-------------------------------+
|                   | General (FK sampling) | Grasp (geometric IK sweep)    |
+-------------------+-----------------------+-------------------------------+
| Radius (XY plane) | 0.005 - 0.546 m       | 0.054 - 0.311 m (with margin) |
| Z height          | -0.186 - 0.489 m      | -0.209 - 0.074 m (with margin)|
+-------------------+-----------------------+-------------------------------+
```

Raw (no margin): R = [0.040, 0.325] m, Z = [-0.225, 0.090] m

**Important:** The grasp workspace boundary is curved — at smaller radii, the
usable Z range shrinks. The bounds above are the outer rectangular envelope.
Positions that pass the pre-check may still fail geometric IK. The pre-check
is a fast first filter; geometric IK is the true arbiter.

### Rejection Chain

When "Move to Grab" is pressed, both the approach and final poses are validated
before the arm moves:

```
1. Ground plane check     z > ground_z?             fast, O(1)
2. Workspace pre-check    r and z within bounds?     fast, O(1)
3. Geometric IK           solution exists?           ~100 us
4. Collision check        solution collision-free?   MoveIt service call
   |
   +-- If ANY stage of ANY pose fails -> reject, arm does not move
```

### Regenerating Bounds

```bash
ros2 run so_arm101_control compute_workspace
# or
python3 compute_workspace.py --samples 200000
```

Both the general workspace and grasp workspace are written to
`workspace_bounds.yaml`.

## Gripper Geometry

The jaw gap follows a linear model derived from STL mesh analysis
(`calibrate_jaw.py`):

```
jaw_gap(m) = BASELINE_JAW_GAP + JAW_GAP_RATE * gripper_joint_angle(rad)
```

```
BASELINE_JAW_GAP = 0.0190 m      (19mm gap at angle=0 - jaws do NOT touch)
JAW_GAP_RATE     = 0.0749 m/rad   (74.9mm per radian)
```

To convert a desired gap to a joint angle:

```
angle = (desired_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
```

Regenerate: `ros2 run so_arm101_control calibrate_jaw`

## Grasp Tab Variables

### 1. Cross-Axis Grasp (checkbox)

Controls which object axis the jaws close across.

```
OFF (default): grip_width = min(sx, sy)   jaws close across short axis
ON:            grip_width = max(sx, sy)   jaws close across long axis, yaw += 90 deg
```

Affects: gripper orientation (wrist_roll), IK offset direction, jaw angles.

### 2. TCP Clearance (spinbox, mm, default=1.0)

Extra IK offset beyond `grip_width/2` to account for jaw material past the
TCP point. Shifts the IK target from the object center by:

```
offset = grip_width/2 + tcp_clearance
direction = (-sin(obj_yaw), cos(obj_yaw))   [fixed-jaw direction]
```

- `tcp_clearance = 0` -> fixed jaw tip exactly at object edge
- `tcp_clearance = 1` -> fixed jaw tip 1mm away from object surface

Affects: arm joint angles (all 5 DOF). Does **not** move the jaw.

Toggle: "TCP offset" checkbox next to the BBox topic field.
When unchecked, offset = 0 and TCP goes to the object center.

### 3. Open Clearance (spinbox, mm, default=5.0)

Extra jaw gap when opening, beyond the symmetric baseline.

```
open_gap   = grip_width + 2*tcp_clearance + open_clearance
open_angle = (open_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
```

Affects: `gripper_joint` only. Arm stays put.

### 4. Close Clearance (spinbox, mm, default=0.0)

Extra jaw gap when closing, beyond the symmetric baseline.

```
close_gap   = grip_width + 2*tcp_clearance + close_clearance
close_angle = (close_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
```

- `close_clearance =  0` -> symmetric gap (tcp_clearance on each side)
- `close_clearance = -1` -> 1mm tighter than symmetric (squeeze)
- `close_clearance = +1` -> 1mm looser than symmetric

Why default is 0: with `tcp_clearance=1mm`, `close_clearance=0` gives 1mm gap
on each side of the object (fixed jaw at 1mm from TCP clearance, moving jaw at
1mm from the symmetric formula).

Affects: `gripper_joint` only. Arm stays put.

### 5. Object Z (spinbox, m, default=0.0)

Override the detected object Z height. When non-zero, this value replaces the
Z from the pose topic.

- `object_z = 0.000` -> use detected Z from topic
- `object_z = 0.008` -> force grasp height to 8mm

### 6. Approach Height (spinbox, m, default=0.05)

Height above the target Z for the first IK stage.

- `approach_height = 0` -> single-stage move directly to target
- `approach_height = 0.05` -> two-stage: approach at +50mm, then descend

### Variable Summary

```
+------------------+-----------+-----------+---------+
| Variable         | Moves arm | Moves jaw | Default |
+------------------+-----------+-----------+---------+
| Cross-axis       | Yes (yaw) | Yes (w)   | OFF     |
| TCP clearance    | Yes (IK)  | No        | 1 mm    |
| TCP offset toggle| Yes (IK)  | No        | ON      |
| Open clearance   | No        | Yes       | 5 mm    |
| Close clearance  | No        | Yes       | 0 mm    |
| Object Z         | Yes (IK)  | No        | 0 (auto)|
| Approach height  | Yes (IK)  | No        | 50 mm   |
+------------------+-----------+-----------+---------+
```

## Move to Grab — Execution Stages

When "Move to Grab" is pressed:

### 1. Preprocessing

1. Extract object yaw from quaternion: `atan2(2*qw*qz, 1-2*qz^2)`
2. If cross-axis: add 90 deg to yaw
3. Normalize yaw: pick `yaw` or `yaw +/- pi` to keep `wrist_roll` within limits
   (jaws are symmetric so `yaw = yaw + pi` for grasping)
4. Compute jaw offset: shift IK target by `grip_width/2 + tcp_clearance`
   from object center along the fixed-jaw direction
5. Compute grasp quaternion: `RPY(0, 90, pan + obj_yaw)` — gripper straight down

### 2. Pre-validation (both stages)

Both the approach pose and final pose are validated **before** the arm moves:
- Workspace pre-check (ground plane, radial bounds, Z bounds)
- Geometric IK (solution exists?)
- Collision check via MoveIt (solution collision-free?)

If either stage fails at any check, the grasp is rejected and the arm stays put.

### 3. Execution

If both stages pass validation:
1. Execute approach trajectory (arm moves to approach height)
2. On completion callback, execute final descent trajectory (pre-validated joints)
3. Each stage sends a `FollowJointTrajectory` action to `ros2_control`

## Gripper Buttons

### Grasp Tab

```
+---------------+-----------------------------------------------------------+
| Button        | Behavior                                                  |
+---------------+-----------------------------------------------------------+
| Grasp Open    | Opens jaw to object-aware angle (requires object selected)|
| Grasp Close   | Closes jaw to object-aware angle (requires object selected|
| Open          | Opens jaw to range spinbox upper value                    |
| Close         | Closes jaw to range spinbox lower value                   |
+---------------+-----------------------------------------------------------+
```

### FK/IK Tab

```
+---------------+-------------------------------------------+
| Button        | Behavior                                  |
+---------------+-------------------------------------------+
| Open          | Opens jaw to full joint limit (100 deg)   |
| Close         | Closes jaw to full joint limit (-10 deg)  |
+---------------+-------------------------------------------+
```

## Debug Services

All buttons are exposed as `std_srvs/Trigger` services under
`/so_arm101_control_gui/`:

```
+---------------------------+---------------------------------------------------+
| Service                   | Description                                       |
+---------------------------+---------------------------------------------------+
| grasp_refresh             | Refresh detected objects list                     |
| grasp_select              | Select object (set ik_target param first)         |
| grasp_move                | Move to Grab (pre-validates both stages)          |
| grasp_reset               | Home position (gripper down, all joints zero)     |
| gripper_open_for_object   | Grasp Open (object-aware)                         |
| gripper_close_for_object  | Grasp Close (object-aware)                        |
| gripper_open_range        | Open (range spinbox value)                        |
| gripper_close_range       | Close (range spinbox value)                       |
| gripper_open              | Open (full joint limit)                           |
| gripper_close             | Close (full joint limit)                          |
| set_jaw_open_clearance    | Set open clearance (jaw_open_clearance_mm param)  |
| set_jaw_close_clearance   | Set close clearance (jaw_close_clearance_mm param)|
| set_tcp_clearance         | Set TCP clearance (tcp_clearance_mm param)        |
| check_grasp_reachable     | Check if selected object is in grasp workspace    |
| get_tcp_pose              | Read TCP link pose via TF2                        |
| get_ee_pose               | Read gripper link pose from GUI labels            |
+---------------------------+---------------------------------------------------+
```

### Example: Set Close Clearance and Test

```bash
ros2 param set /so_arm101_control_gui jaw_close_clearance_mm -- -1.0
ros2 service call /so_arm101_control_gui/set_jaw_close_clearance std_srvs/srv/Trigger
ros2 service call /so_arm101_control_gui/gripper_close_for_object std_srvs/srv/Trigger
```

### Example: Check Reachability Before Grasp

```bash
ros2 param set /so_arm101_control_gui ik_target "blue_2x4"
ros2 service call /so_arm101_control_gui/grasp_select std_srvs/srv/Trigger
ros2 service call /so_arm101_control_gui/check_grasp_reachable std_srvs/srv/Trigger
```
