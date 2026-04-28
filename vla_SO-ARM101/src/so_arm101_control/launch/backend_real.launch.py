"""Real-hardware backend for the SO-ARM101 stack.

Brings up the producers of the topic contract control_gui consumes,
sourced from the physical robot + USB cameras + ArUco/YOLOE perception:

  * jointstatereader        — Feetech serial bridge.
                              accept_command_input=True so external
                              setpoints on /joint_commands write to the
                              follower; mirror_to_follower=False so leader
                              USB teleop is OFF (the planner drives, not
                              a leader arm).
                              Publishes /joint_states (follower position).
  * real_cameras.launch.py  — wrist + top USB cameras as
                              /wrist_camera, /top_camera (so_arm101_bringup).
  * localize_aruco          — ArUco detection on /top_camera → /drop_poses
                              (cup marker frames).
  * localize_yoloe          — YOLOE-based lego detection → /objects_poses_real
                              + /objects_bbox_real (TFMessage + String JSON).

  * trajectory_shim       — hosts /arm_controller/follow_joint_trajectory
                              and /gripper_controller/follow_joint_trajectory
                              action servers (matching MoveIt's
                              moveit_controllers.yaml). Interpolates
                              received trajectories at 20 Hz on
                              /joint_commands; jointstatereader writes
                              those to the follower servos. Velocity
                              scaling via ``velocity_scale_real`` lets
                              first-contact tests run slow.

Pass ``read_only:=true`` to flip jointstatereader to observation-only mode
(accept_command_input=false, publish_commands=false). First-contact
bringup uses this to verify the read path before granting write authority.

Usage (typically via stack.launch.py mode:=real):
  ros2 launch so_arm101_control backend_real.launch.py \\
      leader_port:=/dev/ttyACM1 follower_port:=/dev/ttyACM0 \\
      wrist_idx:=0 top_idx:=1
  # First-contact (no writes to the follower):
  ros2 launch so_arm101_control backend_real.launch.py \\
      follower_port:=/dev/ttyACM0 read_only:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _resolve_cameras(context, *args, **kwargs):
    """Resolve wrist camera index via signature match; skip if missing.

    cv2.VideoCapture indices on macOS aren't stable — they include the
    built-in FaceTime HD camera + Continuity Camera in alphabetical order
    by uniqueID. Hardcoding `wrist_idx=0 top_idx=1` opens the wrong device
    on any non-trivial rig (e.g. top_idx=1 = FaceTime HD on a single-USB
    setup). This function probes the AVFoundation device list at launch
    time and resolves the wrist camera by USB vendor/product/model match
    from aruco_camera_localizer/config/robot_config.yaml.

    Behavior:
      - auto_detect:=true (default) + wrist_idx unset → use signature.
      - auto_detect:=true but signature lookup fails → skip cameras +
        ArUco/YOLOE entirely with a clear log line. NEVER opens the
        FaceTime HD camera by accident.
      - User passed explicit wrist_idx → respected (no probe).
      - enable_top default is now 'false' for real mode (single-cam is
        the documented real-hw default; opt-in for 2-cam rigs).
    """
    bringup_pkg = get_package_share_directory('so_arm101_bringup')
    auto_detect = LaunchConfiguration('auto_detect').perform(context).lower() == 'true'
    wrist_idx_str = LaunchConfiguration('wrist_idx').perform(context)
    top_idx_str = LaunchConfiguration('top_idx').perform(context)
    enable_top_str = LaunchConfiguration('enable_top').perform(context).lower()

    resolved_wrist = wrist_idx_str
    skip_cameras = False
    detect_msg = ''

    if auto_detect and wrist_idx_str in ('', '-1', 'auto'):
        try:
            from aruco_camera_localizer.config_loader import get_config
            from aruco_camera_localizer.find_camera import find_camera_by_signature
            sig = get_config().get_camera_signature()
            if not sig:
                detect_msg = (
                    '[backend_real] auto_detect=true but no camera.signature '
                    'in robot_config.yaml — falling back to wrist_idx=0.')
                resolved_wrist = '0'
            else:
                idx = find_camera_by_signature(sig)
                if idx is None:
                    skip_cameras = True
                    detect_msg = (
                        '[backend_real] wrist camera (signature='
                        f'{sig}) NOT FOUND. Skipping camera_publisher + '
                        'localize_aruco + localize_yoloe. Plug the wrist '
                        'USB camera in and restart, or pass '
                        'auto_detect:=false wrist_idx:=N to override.')
                else:
                    resolved_wrist = str(idx)
                    detect_msg = (
                        f'[backend_real] wrist camera resolved: cv2 idx={idx} '
                        f'(matched {sig.get("model_pattern", "?")} '
                        f'vid={sig.get("vendor_id")} pid={sig.get("product_id")})')
        except ImportError as e:
            detect_msg = (
                f'[backend_real] auto-detect unavailable ({e}); '
                f'using wrist_idx={wrist_idx_str or "0"}.')
            if not resolved_wrist or resolved_wrist == '-1':
                resolved_wrist = '0'

    actions = []
    if detect_msg:
        actions.append(LogInfo(msg=detect_msg))

    if skip_cameras:
        # No wrist cam → skip BOTH cameras AND the perception nodes that
        # subscribe to them. Otherwise localize_aruco/yoloe spin forever
        # waiting for an image topic that never publishes.
        return actions

    # Resolved (or user-overridden) → include real_cameras.launch.py
    # with the right indices.
    real_cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'real_cameras.launch.py')),
        launch_arguments={
            'wrist_idx': resolved_wrist,
            'top_idx': top_idx_str,
            'enable_top': enable_top_str,
        }.items(),
    )
    actions.append(real_cameras_launch)

    # Perception nodes (ArUco + YOLOE) only make sense with a real camera.
    # Co-locating them with real_cameras gating gives one switch for the
    # whole "real perception is up" condition.
    # Both perception nodes default-subscribe to /camera/image_raw — but
    # our wrist camera publishes to /wrist_camera. Pass --camera-topic
    # via arguments (the nodes use argparse, not just topic remap, so
    # ROS-level remapping doesn't reach them — they read their config
    # from CLI flags).
    # Topic-name contract for real mode:
    #   /objects_poses_real  ← yoloe lego detections (matches sim's
    #                          /objects_poses_sim from sim_ground_truth)
    #   /objects_bbox_real   ← bbox catalog (sx/sy/sz meters per name)
    #   /drop_poses          ← aruco cup-marker detections (no _real
    #                          suffix; same name in sim + real)
    #   /aruco_poses         ← raw marker poses (debug/diagnostic)
    # The bridge's defaults are the un-suffixed variants (/objects_poses,
    # /objects_bbox) — control_gui's Grasp tab subscribes to the _real
    # suffix in real mode, so without these remaps perception runs but
    # never reaches the Grasp tab. Discovered 2026-04-27 while verifying
    # the yoloe→aruco merged-window fix; perception was healthy
    # (132 Hz on /objects_poses, 2 Hz on /objects_bbox) but the Grasp
    # tab was silent.
    real_topic_remap = {
        'objects_poses_topic': '/objects_poses_real',
        'objects_bbox_topic': '/objects_bbox_real',
    }
    actions.append(Node(
        package='aruco_camera_localizer',
        executable='localize_aruco',
        name='localize_aruco',
        output='screen',
        arguments=['--camera-topic', '/wrist_camera'],
        parameters=[{'use_sim_time': False, **real_topic_remap}],
    ))
    actions.append(Node(
        package='aruco_camera_localizer',
        executable='localize_yoloe',
        name='localize_yoloe',
        output='screen',
        # --headless: yoloe doesn't open its own cv2 viewer.
        #
        # annotated_image_topic:='yoloe_annotated' routes yoloe's rich
        # overlay frame (bbox + 3D cuboid wireframe + orientation axes)
        # to the /yoloe_annotated topic. localize_aruco subscribes there
        # via LocalizerBridge._yoloe_annotated_cb and uses the cached
        # frame as the BASE for its "Merged Detection" window when
        # /aruco_viewer/yoloe_overlay_enable is true (default). Without
        # this remap, yoloe publishes to the bridge's default
        # 'camera_annotated' and aruco's merged window can't find it,
        # so the user only sees aruco overlays — that's the bug we're
        # fixing here.
        #
        # The two SetBool services (/aruco_viewer/aruco_overlay_enable +
        # /aruco_viewer/yoloe_overlay_enable) are advertised by aruco
        # only (yoloe passes advertise_overlay_services=False inside
        # start_ros_node), so no service-name collision. Live-toggle
        # via the "Detector Overlays" checkboxes in control_gui's
        # Settings tab.
        arguments=['--camera-topic', '/wrist_camera', '--headless'],
        parameters=[{
            'use_sim_time': False,
            'annotated_image_topic': 'yoloe_annotated',
            # Both bridge processes (aruco + yoloe) own a publisher for
            # objects_poses / objects_bbox — must remap on BOTH otherwise
            # yoloe keeps publishing to the un-suffixed default, creating
            # a parallel stream the Grasp tab never sees. See real_topic_remap
            # comment above the localize_aruco Node.
            **real_topic_remap,
        }],
    ))
    return actions


def generate_launch_description():
    leader_port = LaunchConfiguration('leader_port')
    follower_port = LaunchConfiguration('follower_port')
    read_only = LaunchConfiguration('read_only')
    velocity_scale_real = LaunchConfiguration('velocity_scale_real')

    # --- Real arm: Feetech serial bridge ---
    # Two configurations chosen by ``read_only``:
    #   read_only:=false (default) — topic-driven write mode (mode 2 in
    #     joint_state_reader.py): trajectory_shim / control_gui /
    #     a policy publishes JointState setpoints to /joint_commands and
    #     this node writes them to the follower servos.
    #   read_only:=true — observation-only (mode 3): /joint_states is
    #     published from the follower's measured position; nothing writes.
    #     Use this for first-contact bringup (RViz mirror, sanity-check
    #     state path) before granting any write authority.
    jointstatereader_write = Node(
        package='jointstatereader',
        executable='joint_state_reader',
        name='joint_state_reader',
        output='screen',
        parameters=[{
            'leader_port': leader_port,
            'follower_port': follower_port,
            'mirror_to_follower': False,
            'accept_command_input': True,
            'publish_source': 'follower',
            'publish_commands': True,
            'use_sim_time': False,
        }],
        condition=UnlessCondition(read_only),
    )
    jointstatereader_observe = Node(
        package='jointstatereader',
        executable='joint_state_reader',
        name='joint_state_reader',
        output='screen',
        parameters=[{
            'leader_port': leader_port,
            'follower_port': follower_port,
            'mirror_to_follower': False,
            'accept_command_input': False,
            'publish_source': 'follower',
            'publish_commands': False,
            'use_sim_time': False,
        }],
        condition=IfCondition(read_only),
    )

    # --- Real cameras + perception ---
    # Camera resolution + ArUco/YOLOE inclusion are decided at launch time
    # by _resolve_cameras (signature-based wrist detection — see top of
    # file). When the wrist USB cam isn't plugged in, the entire camera
    # + perception subgraph is skipped with a clear log line, so we
    # never open the wrong AVFoundation device (FaceTime HD on macOS
    # single-USB rigs).
    cameras_and_perception = OpaqueFunction(function=_resolve_cameras)

    # --- Trajectory shim: MoveIt FollowJointTrajectory → /joint_commands ---
    # Only spawned when read_only=false (no point in a write-path executor
    # when jointstatereader is in observation-only mode). Velocity scaling
    # defaults to 0.25 (4x slower than MoveIt's planned timing) so the
    # first real-arm motions are deliberate; raise toward 1.0 once you
    # trust the path.
    trajectory_shim = Node(
        package='so_arm101_control',
        executable='trajectory_shim',
        name='trajectory_shim',
        output='screen',
        parameters=[{
            'velocity_scale': velocity_scale_real,
            'rate_hz': 20.0,
            'use_sim_time': False,
            # Route through gravity_comp: shim writes /joint_targets, then
            # gravity_comp_node applies K_calib feed-forward and republishes
            # to /joint_commands. Without this, K_calib calibration is a
            # no-op and the arm sags by the full uncompensated amount.
            'joint_commands_topic': '/joint_targets',
        }],
        condition=UnlessCondition(read_only),
    )

    # --- Gravity compensation node (Layer 2 + Layer 3) ---
    # Subscribes /joint_targets, applies model-based gravity FF (Pinocchio
    # RNEA on the URDF) + adaptive integrator, publishes /joint_commands.
    # Users that want compensation publish to /joint_targets; raw
    # /joint_commands publishers (existing code) bypass it.
    #
    # Skipped when read_only=true (no write path) or enable_gravity_comp=false
    # (e.g. for diagnostics where you want to see uncompensated behavior).
    enable_gravity_comp = LaunchConfiguration('enable_gravity_comp')
    gravity_comp_node = Node(
        package='so_arm101_control',
        executable='gravity_comp_node',
        name='gravity_comp',
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", read_only, "' == 'false' and '",
            enable_gravity_comp, "' == 'true'",
        ])),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'leader_port', default_value='/dev/ttyACM1',
            description='Feetech leader serial port (ignored; teleop is OFF in real-mode).'),
        DeclareLaunchArgument(
            'follower_port', default_value='/dev/ttyACM0',
            description='Feetech follower serial port — the arm we drive.'),
        DeclareLaunchArgument(
            'auto_detect', default_value='true',
            description=(
                'When true (default), resolve wrist_idx via USB signature '
                'match against aruco_camera_localizer/config/robot_config.yaml. '
                'Set to false to use the literal wrist_idx value.')),
        DeclareLaunchArgument(
            'wrist_idx', default_value='-1',
            description=(
                'cv2.VideoCapture index for the wrist camera. -1 (default) '
                'lets auto_detect resolve it via signature. Pass an explicit '
                'integer to override.')),
        DeclareLaunchArgument(
            'top_idx', default_value='1',
            description='cv2.VideoCapture index for the overhead camera.'),
        DeclareLaunchArgument(
            'enable_top', default_value='false',
            description=(
                'Launch the top camera publisher. Default false because the '
                'documented real-hw rig is single-cam (wrist only). Pass '
                'enable_top:=true top_idx:=N for 2-cam setups.')),
        DeclareLaunchArgument(
            'read_only', default_value='false',
            description=(
                'When true, run jointstatereader in observation-only mode '
                '(accept_command_input=false, publish_commands=false) and '
                'skip the trajectory_shim. Use for first-contact bringup '
                'before granting write authority.')),
        DeclareLaunchArgument(
            'velocity_scale_real', default_value='1.0',
            description=(
                'trajectory_shim velocity multiplier on time_from_start. '
                'Default 1.0 plays MoveIt-planned trajectories at the '
                'planned speed (no time-stretching). Raise above 1.0 to '
                'speed up beyond MoveIt plan, lower for first-contact '
                'safety. Live-tunable via Settings tab spinbox or '
                '`ros2 param set /trajectory_shim velocity_scale <val>`.')),
        DeclareLaunchArgument(
            'enable_gravity_comp', default_value='true',
            description=(
                'Layer 2 (gravity FF) + Layer 3 (adaptive integrator) on '
                '/joint_targets → /joint_commands. Loads K_calib from '
                '~/.cache/so_arm101/gravity_comp/k_calib.json. Set false '
                'to bypass for diagnostics (publishers must use '
                '/joint_commands directly).')),

        jointstatereader_write,
        jointstatereader_observe,
        cameras_and_perception,
        trajectory_shim,
        gravity_comp_node,
    ])
