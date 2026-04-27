# Gazebo Backend on Linux — Known Limitation

> **Status (2026-04-27)**: Gazebo simulation cannot drive arm motion on the local
> Linux setup. Gripper works, world loads, control GUI starts. Documented for
> future revisit; not blocking the Isaac Sim sim path.

## TL;DR

Running `ros2 launch so_arm101_control gazebo.launch.py` on Linux brings
up Gazebo Ignition, the world, the `gz_ros2_control` plugin, MoveIt, and
the control GUI cleanly — but `arm_controller` fails to activate.
`grasp_home` (and any other arm motion service) fails with
`arm action server rejected the goal`.

Root cause: a mismatch between local commit `eb13c10` (Phase 9 PD
feed-forward — adds `velocity` command interface to `arm_controller`'s
yaml) and the version of `gz_ros2_control/GazeboSimSystem` currently
built on Linux (`~/ros2_ws/install/gz_ros2_control/`), which appears
not to support multi-command-interface joints in the URDF.

This is NOT a regression introduced by the upstream-merge commit
`a01e646` — `eb13c10` predates the merge and the Gazebo path was simply
never re-tested after Phase 9.

## Symptoms

```
[controller_manager]: Could not switch controllers since prepare command
  mode switch was rejected.
[spawner_arm_controller]: Failed to activate controller : arm_controller
  [exit code 1]
```

Controller state after launch:

```
ros2 control list_controllers:
  joint_state_broadcaster   active
  arm_controller             inactive    ← cannot move arm
  gripper_controller         active      (yaml is position-only — matches xacro)
```

Functional probe:

```
ros2 service call /so_arm101_control_gui/grasp_home std_srvs/srv/Trigger
  → success=False, message='_cmd_grasp_home: arm action server rejected the goal'
```

## Root cause

`src/so_arm101_moveit_config/config/ros2_controllers.yaml`:

```yaml
arm_controller:
  command_interfaces:
    - position
    - velocity   # ← added by commit eb13c10 for Isaac Sim PD feed-forward
  state_interfaces:
    - position
    - velocity
```

`src/so_arm101_description/urdf/so_arm101.gazebo.xacro`:

```xml
<joint name="shoulder_pan">
  <command_interface name="position"/>   <!-- only position -->
  <state_interface name="position"/>
  <state_interface name="velocity"/>
</joint>
```

The yaml asks for both `position` and `velocity` command interfaces, but
the URDF declares only `position`, so
`controller_manager.prepareCommandModeSwitch()` rejects activation. With
five arm joints in this state, `arm_controller` never goes active.

## What was tried (and why it failed)

Adding `<command_interface name="velocity"/>` to all five arm joints in
`so_arm101.gazebo.xacro` did NOT fix it on Linux. The result:

- `ros2 control list_hardware_interfaces` returned **empty** lists for
  both command and state interfaces.
- All three controllers (joint_state_broadcaster, arm_controller,
  gripper_controller) failed to activate — including gripper, which
  worked before the change.

Interpretation: the Linux-side `gz_ros2_control/GazeboSimSystem` (built
from `~/ros2_ws/install/gz_ros2_control/`, ROS2 Humble / Ignition
Fortress era) does not support multi-command-interface joints. When it
encounters a joint with more than one `<command_interface>`, it appears
to silently fail URDF parsing and expose zero interfaces for that joint
(and possibly bail out for the rest of the system).

The xacro change has been reverted; current state matches the merge
commit `a01e646`.

## Why the Mac path probably works (untested)

`mac-env/pixi.toml` pulls a substantially newer stack via RoboStack:

| Component | Linux (apt + locally built) | Mac (pixi/RoboStack jazzy) |
|---|---|---|
| ROS2 distro | Humble | **Jazzy** |
| Gazebo | Ignition Fortress | **Harmonic** |
| `gz_ros2_control` | locally-built older | **v1.2.17** from `robostack-jazzy/osx-arm64` |

Multi-command-interface support was actively improved in the Jazzy-era
1.2.x range of `gz_ros2_control`. The Mac path most likely handles
`position + velocity` claims correctly — but this is **not verified**.
To confirm on Mac: run `gazebo.launch.py`, then
`ros2 control list_controllers` and check whether `arm_controller` is
`active`.

## Workaround paths (not implemented)

| Approach | Pros | Cons |
|---|---|---|
| **A**. Drop `velocity` from `arm_controller` yaml | One-line fix, makes Linux Gazebo work | Breaks Phase 9 PD feed-forward on Isaac Sim → tracking lag returns (~10° at peak velocity) → cup padding (5%) likely insufficient → drop_sweep collisions return |
| **B**. Platform-conditional yaml (Gazebo variant: position-only; Isaac variant: position+velocity) | Both backends work | Two yamls to maintain; launch wiring needs to select the right one based on backend |
| **C**. Upgrade Linux's `gz_ros2_control` to a Jazzy-era version | Same single yaml works everywhere | Painful — would need to either build Jazzy on Humble (ABI mismatch) or upgrade the whole ROS2 distro |
| **D**. Accept Linux Gazebo as broken; use Isaac Sim only on Linux | Zero work; today's actual practice | Lose the option of Gazebo on Linux; can't onboard a contributor without an Isaac Sim license unless they're on Mac |

The user's current preference is **D** — Isaac Sim is the primary working
backend on Linux. Revisit when one of the trigger conditions below applies.

## When to revisit

Most likely triggers:

1. Collecting lerobot training datasets on Linux and wanting Gazebo
   (different rendering, different physics) instead of Isaac Sim.
2. Onboarding a Linux contributor who can't get an Isaac Sim license.
3. Verifying the Mac path actually works, then wanting feature parity on
   Linux.
4. The `eb13c10` velocity command-interface optimization stops being
   load-bearing on Isaac Sim — at that point, dropping it from the yaml
   (workaround **A**) becomes free.

## Verification status (2026-04-27)

- Isaac Sim path: ✅ end-to-end PASS via
  `scripts/test_qs_cycle.sh red_2x3` against the merged code (full
  pick-and-drop cycle, tier-1 + tier-2 deterministic planner, pan-lock,
  `_attached_lego_tcp_offset` capture).
- Gazebo on Linux: ❌ broken (this document).
- Gazebo on Mac: untested.
- Real hardware: untested (separate scope).

## Rollback anchors (in case future investigation needs to revert)

- Pre-merge git tag in `Exploring-VLAs/`: `pre-merge-snapshot` (points to
  `f131e48`).
- Quarantined merge originals (base / local / upstream / merged-with-markers
  for every conflicted file plus the auto-merged `control.launch.py`):
  `~/Documents/merge-quarantine/vla-SO-ARM101-2026-04-27/`.
- Decision log for each merge resolution:
  `~/Documents/merge-quarantine/vla-SO-ARM101-2026-04-27/DECISIONS.md`.
