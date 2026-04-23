"""Bring up sim ground-truth pose + bbox publishing.

Runs a single node: ``sim_ground_truth/ground_truth_publisher``. The node
subscribes to the Gazebo world pose info topic directly via ``gz.transport13``
and republishes filtered poses + bbox JSON on standard ROS2 topics.

``ros_gz_bridge`` is **not** used here: its ``gz.msgs.Pose_V → tf2_msgs/TFMessage``
conversion drops ``pose.name`` and ``pose.id``, making the bridged messages
unusable for per-object filtering.

Example::

    ros2 launch sim_ground_truth ground_truth.launch.py \
        world_name:=so_arm101_lego_world
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    world_name = LaunchConfiguration('world_name')
    objects_yaml = LaunchConfiguration('objects_yaml')
    poses_rate = LaunchConfiguration('poses_rate_hz')
    bbox_rate = LaunchConfiguration('bbox_rate_hz')
    drop_poses_rate = LaunchConfiguration('drop_poses_rate_hz')
    drop_poses_topic = LaunchConfiguration('drop_poses_topic')
    log_unknown = LaunchConfiguration('log_unknown_entities')

    pkg_share = get_package_share_directory('sim_ground_truth')
    default_yaml = os.path.join(pkg_share, 'config', 'lego_world_objects.yaml')

    declarations = [
        DeclareLaunchArgument(
            'world_name', default_value='so_arm101_lego_world',
            description='Gazebo world name (<world name="..."> in the loaded '
                        'SDF, not the SDF filename).'),
        DeclareLaunchArgument('objects_yaml', default_value=default_yaml,
                              description='YAML catalog of objects + bbox sizes.'),
        DeclareLaunchArgument('poses_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('bbox_rate_hz', default_value='1.0'),
        DeclareLaunchArgument('drop_poses_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('drop_poses_topic', default_value='/drop_poses',
                              description='Topic for cup ArUco-marker poses '
                                          '(TFMessage). Set to empty/disable via '
                                          'drop_poses_rate_hz:=0.'),
        DeclareLaunchArgument('log_unknown_entities', default_value='false',
                              description='Log first occurrence of each gz entity '
                                          'name not in the catalog (debug aid).'),
    ]

    filter_node = Node(
        package='sim_ground_truth',
        executable='ground_truth_publisher',
        name='sim_ground_truth_publisher',
        output='screen',
        parameters=[{
            'world_name': world_name,
            'objects_yaml': objects_yaml,
            'poses_rate_hz': poses_rate,
            'bbox_rate_hz': bbox_rate,
            'drop_poses_rate_hz': drop_poses_rate,
            'drop_poses_topic': drop_poses_topic,
            'log_unknown_entities': log_unknown,
        }],
    )

    return LaunchDescription(declarations + [filter_node])
