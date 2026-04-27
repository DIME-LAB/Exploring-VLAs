"""Gazebo Harmonic backend for the SO-ARM101 stack.

Launches everything that produces the topic contract control_gui consumes,
sourced from a Gazebo simulation:

  * gz sim                  (server with -s; optional GUI client with -g)
  * parameter_bridge        (/clock + camera images: gz↔ROS)
  * gz_spawn_entity         (spawn the URDF into the running world)
  * controller spawners     (joint_state_broadcaster → arm_controller →
                             gripper_controller, sequenced via OnProcessExit)
  * sim_ground_truth        (gz.transport13 → /objects_poses_sim,
                             /objects_bbox_sim, /drop_poses)

Pairs with core.launch.py — that one supplies robot_state_publisher,
move_group, control_gui, etc. Together they reproduce the original
gazebo.launch.py behaviour, but split so a different backend can replace
this file without touching core.

Usage (typically via stack.launch.py mode:=gazebo):
  ros2 launch so_arm101_control backend_gazebo.launch.py headless:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    desc_pkg = get_package_share_directory('so_arm101_description')
    world_file = os.path.join(desc_pkg, 'worlds', 'lego_world.sdf')

    # Gazebo needs to find package:// meshes via GZ_SIM_RESOURCE_PATH and
    # the gz_ros2_control plugin via GZ_SIM_SYSTEM_PLUGIN_PATH. Same
    # paths as the original gazebo.launch.py.
    install_share_parent = os.path.dirname(desc_pkg)
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    pixi_lib = os.path.join(conda_prefix, 'lib') if conda_prefix else ''

    headless = LaunchConfiguration('headless')

    # On macOS, gz sim can't run server+GUI in one process (gz-sim#44).
    # Always start the server with -s; if not headless, fork a -g client.
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ]),
        launch_arguments={'gz_args': '-r -s ' + world_file}.items(),
    )

    gz_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen',
        condition=UnlessCondition(headless),
        additional_env={
            'GZ_SIM_RESOURCE_PATH': install_share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
            'GZ_SIM_SYSTEM_PLUGIN_PATH': pixi_lib + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', ''),
        },
    )

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

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            '/wrist_camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/top_camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/top_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    # Sequenced controller spawn: each one waits for the previous to exit
    # (the spawner CLI exits after Configured+activated). On macOS the
    # ros2 daemon hangs, so we use the spawner Node (DDS-direct) instead.
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '120'],
        output='screen',
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager-timeout', '120'],
        output='screen',
    )

    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager-timeout', '120'],
        output='screen',
    )

    # Sim ground truth — mirrors aruco_camera_localizer's topic contract
    # so control_gui consumes the same topics in sim as in real.
    sim_ground_truth = Node(
        package='sim_ground_truth',
        executable='ground_truth_publisher',
        name='ground_truth_publisher',
        output='screen',
        parameters=[{
            'world_name': 'so_arm101_lego_world',
            'use_sim_time': True,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run Gazebo without the GUI client (server only).'),

        # 0. Gazebo discovery paths.
        SetEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=install_share_parent + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')),
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=install_share_parent + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')),
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=pixi_lib + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')),

        # 1. Gazebo + bridge.
        gz_server,
        gz_gui,
        bridge,

        # 2. Spawn URDF (consumes /robot_description from core.launch.py's RSP).
        gz_spawn_entity,

        # 3. Sequential controller activation chained off spawn completion.
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

        # 4. Ground truth — fine to start anytime, but waits internally for
        #    gz.transport to publish poses.
        sim_ground_truth,
    ])
