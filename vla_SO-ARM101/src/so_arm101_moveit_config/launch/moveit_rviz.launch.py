from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("so_arm101", package_name="so_arm101_moveit_config")
        .to_moveit_configs()
    )

    rviz_config = str(moveit_config.package_path / "config/moveit.rviz")

    # The default generate_moveit_rviz_launch helper does not pass
    # robot_description / robot_description_semantic as parameters, which
    # leaves RViz unable to render the robot or parse the SRDF
    # (no-one else in this stack publishes /robot_description_semantic).
    # Passing them explicitly — same pattern as moveit2_tutorials.
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([rviz_node])
