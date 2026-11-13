# soarm101-dt Extension

Isaac Sim 6.1 extension for the SO-ARM101 digital twin. Its commands are served as `soarm101.*` tools
through the omni-kit-mcp bridge (see `isaac/soarm101_tools`).

## Features

- **SO-ARM101 Robot Control**: Load and configure the SO-ARM101 5-DOF robotic arm with integrated gripper
- **ROS2 Joint Control**: Action graphs for joint state subscription, gripper control, and force/torque publishing
- **Intel RealSense Camera**: Wrist-mounted camera with custom 3D-printed mount
- **Lego Block Objects**: Add, delete, sort, and randomize colored lego blocks (red, green, blue; 2x2, 2x3, 2x4)
- **Scene State Management**: Save and restore object poses to JSON for repeatable experiments
- **Sim-to-Real Sync**: Subscribe to real-world detected poses and mirror them in simulation
- **MCP Integration**: every command in `so_arm101_dt/tool_table.py` is a `soarm101.<name>` tool on the omni-kit-mcp bridge

## MCP Tools

Selected tools (full list: `so_arm101_dt/tool_table.py`, or `python3 -m kit_mcp.client list_tools`):

| Tool | Description |
|------|-------------|
| `quick_start` | Build the whole scene (robot, cameras, legos, cups, publishers) and play |
| `play_scene` | Start/resume the simulation timeline |
| `stop_scene` | Stop the simulation timeline |
| `load_scene` | Initialize scene with physics, ground plane, and frame rate settings |
| `load_robot` | Import SO-ARM101 robot USD with joint drives and gripper physics |
| `setup_action_graph` | Create ROS2 action graph for joint state subscription and clock |
| `setup_force_publisher` | Create ROS2 force/torque publisher for end-effector wrench |
| `setup_wrist_camera_action_graph` | Create ROS2 action graph for wrist camera (OV9732) RGB and camera_info |
| `add_objects` | Add lego block objects to the scene |
| `delete_objects` | Delete all objects and associated action graphs |
| `sort_objects` | Sort blocks by placing same-color blocks in clusters |
| `randomize_object_poses` | Randomize block positions with per-object collision avoidance |
| `randomize_single_object` | Randomize a single named object while keeping others fixed |
| `save_scene_state` | Save current object poses to a timestamped JSON file |
| `restore_scene_state` | Restore object poses from a JSON file |
| `setup_pose_publisher` | Publish object poses to ROS2 topic `objects_poses_sim` |
| `sync_real_poses` | Subscribe to `/objects_poses_real` and update sim poses |

## Calling the tools

Launch with `linux-env/scripts/isaac/isaacsim_launch.sh launch soarm101-dt [--stream]`, which
enables the bridge (port 8767) and loads `soarm101_tools` for that launch. Then:

```bash
PYTHONPATH=<repo>/omni-kit-mcp python3 -m kit_mcp.client soarm101.quick_start
PYTHONPATH=<repo>/omni-kit-mcp python3 -m kit_mcp.client soarm101.randomize_object_poses '{"color":"red"}'
```

A tool returns the handler's result dict on success; a handler that reports a non-success status
or raises becomes a bridge error (`kit_mcp.client.BridgeError`). Arbitrary Python goes through
the bridge's built-in `run_python`.

## Configuration

- Robot USD: `assets/SO-ARM101-USD.usd`
- Lego blocks: `assets/legos/` (per-instance copies are generated into `assets/legos/instances/` at runtime)
- Objects folder: `/World/Objects`
