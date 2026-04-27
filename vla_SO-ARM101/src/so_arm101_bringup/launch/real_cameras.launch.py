"""Launch two camera_publisher instances for real SO-ARM101 hardware.

Matches the sim pipeline's camera topology:

* ``/wrist_camera`` — USB camera mounted on the arm wrist
* ``/top_camera``   — overhead USB camera

Both stream BGR8 at 640x480 @ 30 fps to match Phase 3 sim-side recording.
The so101_ros2 lerobot plugin subscribes to these topic names directly,
so no remapping layer is needed.

Override the cv2 device indices at launch time::

    ros2 launch so_arm101_bringup real_cameras.launch.py \
        wrist_idx:=0 top_idx:=2 enable_top:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    wrist_idx = LaunchConfiguration('wrist_idx')
    top_idx = LaunchConfiguration('top_idx')
    enable_top = LaunchConfiguration('enable_top')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')

    declarations = [
        DeclareLaunchArgument('wrist_idx', default_value='0',
                              description='cv2.VideoCapture index for the wrist camera'),
        DeclareLaunchArgument('top_idx', default_value='1',
                              description='cv2.VideoCapture index for the top camera'),
        DeclareLaunchArgument('enable_top', default_value='true',
                              description='Launch the top camera publisher'),
        DeclareLaunchArgument('width', default_value='640'),
        DeclareLaunchArgument('height', default_value='480'),
        DeclareLaunchArgument('fps', default_value='30.0'),
    ]

    wrist_node = Node(
        package='so_arm101_bringup',
        executable='camera_publisher',
        name='wrist_camera',
        output='screen',
        arguments=[
            '--camera-id', wrist_idx,
            '--publish-topic', '/wrist_camera',
            '--width', width,
            '--height', height,
            '--fps', fps,
            '--frame-id', 'wrist_camera_optical_frame',
        ],
    )

    top_node = Node(
        package='so_arm101_bringup',
        executable='camera_publisher',
        name='top_camera',
        output='screen',
        condition=IfCondition(enable_top),
        arguments=[
            '--camera-id', top_idx,
            '--publish-topic', '/top_camera',
            '--width', width,
            '--height', height,
            '--fps', fps,
            '--frame-id', 'top_camera_optical_frame',
        ],
    )

    return LaunchDescription(declarations + [wrist_node, top_node])
