# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- Ported to Isaac Sim 6.1: declared the deprecated-but-shipped `isaacsim.core.api/prims/utils` and the other
  imported extensions as dependencies; dropped the Python 3.11 rclpy sys.path hack (6.1 bundles rclpy for 3.12)
  and the unused Lula import.
- Commands are served through omni-kit-mcp as `soarm101.*` (tool package `isaac/soarm101_tools`); the tool
  table moved to `so_arm101_dt/tool_table.py`.
- `load_scene`, `load_robot`, `setup_action_graph` handlers await their coroutine (the result now reflects
  completion); `new_stage` awaits frames instead of pumping `app.update()`.

### Added
- Baked ArUco marker meshes (`assets/aruco_markers/meshes/aruco_marker_00{1,2,3}.usda` + textures) are committed;
  the cups match the training images without Blender at runtime. Rebake with Blender 4.x via `SOARM_BLENDER`
  (the bake only runs when a file is missing; without Blender 4.x the extension warns that the cups lack markers).

### Fixed
- Cold-cache `quick_start` freeze in `load_robot`: UJITSO collision cooking disabled at startup
  (ported from isaac-sim-mcp 109363a), PhysX warmup waits after play (5fc3464, b6485d4).
- `randomize_single_object` defaults its X/Y ranges to the `randomize_object_poses` workspace.

### Removed
- The private TCP server on port 8767 and `execute_python_code` (the bridge's `run_python` replaces it).

## [1.2.0] - 2026-03-17

### Added
- USB camera module: `usb_cam_elp.usd` mesh in `assets/cameras/`, converted from URDF STL
- `attach_usb_camera()` method: attaches USB camera mesh + camera prim to camera_mount_link using URDF `usb_camera_joint` transform
- "Attach USB Camera" UI button in Wrist Camera section
- `_get_usb_camera_usd_path()` helper for camera mesh asset loading

### Fixed
- **USD camera chain now matches URDF exactly**: deleted broken `camera_link_joint` (wrong combined transform from URDF converter), added missing `usb_camera_link` physics body (mass=0.001kg) + `usb_camera_joint` + correct `camera_link_joint` with exact URDF values
- **Camera orientation**: `Rx(+90°)*Rz(-90°)` quaternion `(0.5, 0.5, 0.5, -0.5)` correctly maps Isaac Sim -Z rendering axis to physical lens direction with gripper at bottom of frame
- **YOLOE detection accuracy**: 3.2mm avg error (previously 93mm+), matching Gazebo baseline (1.6mm) with identical `opencv_to_camera` calibration — no sim-specific tuning needed

### Changed
- `attach_camera_mount_mesh()` is now mount-only (no longer creates camera prim)
- Camera prim lives at `camera_mount_link/usb_camera/wrist_camera` (was `camera_link/wrist_camera`)
- Camera prim uses three-level structure: unscaled joint Xform → scaled mesh child + camera sibling (prevents scale mangling)
- `setup_wrist_camera_action_graph()` updated to new camera prim path, auto-calls `attach_usb_camera()`
- Quick Start calls both `attach_camera_mount_mesh()` + `attach_usb_camera()`

## [1.1.0] - 2026-02-28

### Added
- Bundled local USD assets: SO-ARM101-USD.usd, lego blocks (red/green/blue 2x2/2x3/2x4), camera_wrist_mount.usd
- Wrist camera section: Setup Camera Action Graph button and `setup_wrist_camera_action_graph` MCP tool
- BBox publisher: `setup_bbox_publisher` MCP tool for object bounding boxes on `objects_bbox_sim`
- `assets/legos/README.md` and `assets/wrist_mounts/` layout for local assets
- `scripts/fix_soarm101_usd.py` to copy and fix original USD (base_delta, camera mount reference)

### Changed
- **Local-only assets**: Robot, legos, camera mount load from `assets/`; no Nucleus fallback (FileNotFoundError if missing)
- Pre-fixed SO-ARM101 USD in `assets/SO-ARM101-USD.usd` (base_delta baked in; skip runtime correction)
- Joint drive params: max force 3 Nm (STS3215 spec), stiffness/damping tuned for real2sim
- **MCP handlers delegate to async methods**: `load_scene`, `load_robot`, `setup_action_graph` now use `run_coroutine()` to call the same code as UI buttons (fixes ArticulationView, camera mass/inertia, simulation context)
- Quick Start: camera mount, lego objects, pose publisher, bbox publisher; removed workspace camera and gripper action graph steps

### Removed
- Gripper action graph (integrated gripper controlled via `/joint_states` joint index 5)
- `setup_gripper_action_graph` MCP tool and UI button

## [1.0.0] - 2026-02-28
- Initial release of SO-ARM101 digital twin extension
- SO-ARM101 5-DOF arm with integrated gripper control
- ROS2 action graphs: joint state subscriber, gripper control, force/torque publisher
- Wrist-mounted Intel RealSense camera with custom mount
- Lego block object management (add, delete, sort, randomize)
- Per-object collision-aware randomization with largest-first placement
- Scene state save/restore with timestamped JSON files
- Sim-to-real object pose synchronization via ROS2
- Object pose publisher to ROS2 topic
- Quick Start UI button for one-click setup
- MCP socket server integration on port 8767
