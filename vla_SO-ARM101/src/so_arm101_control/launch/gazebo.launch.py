# Reference: ~/ros2_ws/src/jetank_description/launch/jetank_gazebo.launch.py
"""
Gazebo (Ignition) simulation launch for SO-ARM101.
Spawns the robot in a Gazebo world with physics, loads ros2_control controllers,
and optionally launches MoveIt + control GUI.

Usage:
  ros2 launch so_arm101_control gazebo.launch.py
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory('so_arm101_description')
    moveit_pkg = get_package_share_directory('so_arm101_moveit_config')

    # Process the Gazebo-specific xacro (includes base URDF + gz_ros2_control)
    xacro_file = os.path.join(desc_pkg, 'urdf', 'so_arm101.gazebo.urdf.xacro')
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    robot_description = doc.toxml()

    world_file = os.path.join(desc_pkg, 'worlds', 'ground_plane.world')

    # Ignition Gazebo needs to find package:// meshes via IGN_GAZEBO_RESOURCE_PATH.
    # The share parent (e.g. .../install/so_arm101_description/share) contains
    # 'so_arm101_description/meshes/', which is what Gazebo looks for.
    install_share_parent = os.path.dirname(desc_pkg)

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # --- Gazebo sim ---
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ]),
        launch_arguments={'gz_args': '-r ' + world_file}.items(),
    )

    # --- Robot State Publisher ---
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    # --- Spawn robot into Gazebo ---
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'so_arm101',
            '-allow_renaming', 'true',
        ],
    )

    # --- Controller loaders (sequential after spawn) ---
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen',
    )

    load_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'arm_controller'],
        output='screen',
    )

    load_gripper_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'gripper_controller'],
        output='screen',
    )

    # --- ros_gz_bridge for /clock + camera ---
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            '/wrist_camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    # --- MoveIt move_group ---
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
    )

    # --- Control GUI ---
    control_gui = Node(
        package='so_arm101_control',
        executable='control_gui',
        name='so_arm101_control_gui',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # --- EE Pose Publisher ---
    ee_pose_publisher = Node(
        package='so_arm101_control',
        executable='ee_pose_publisher',
        name='ee_pose_publisher',
        parameters=[{
            'base_frame': 'base',
            'ee_frame': 'gripper',
            'publish_rate': 10.0,
            'startup_delay': 5.0,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # --- Camera Pose Publisher ---
    camera_pose_publisher = Node(
        package='so_arm101_control',
        executable='camera_pose_publisher',
        name='camera_pose_publisher',
        parameters=[{
            'base_frame': 'base',
            'camera_frame': 'camera_link',
            'publish_rate': 10.0,
            'startup_delay': 5.0,
            'use_sim_time': True,
        }],
        output='screen',
    )

    # --- RViz (via MoveIt launch for full plugin support) ---
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulated clock from Gazebo'),

        # 0. Set resource path so Gazebo can find package:// meshes
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=install_share_parent + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')),

        # 1. Start Gazebo
        gz_sim,

        # 2. Publish robot description
        robot_state_publisher,

        # 3. Spawn robot into Gazebo
        gz_spawn_entity,

        # 4. Sequential controller loading after spawn completes
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_arm_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_arm_controller,
                on_exit=[load_gripper_controller],
            )
        ),

        # 5. Bridge + MoveIt + GUI after controllers are ready
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_gripper_controller,
                on_exit=[bridge, move_group, control_gui, ee_pose_publisher, camera_pose_publisher, rviz],
            )
        ),
    ])
