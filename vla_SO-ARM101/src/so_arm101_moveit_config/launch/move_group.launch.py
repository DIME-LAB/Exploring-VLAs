from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "so_arm101", package_name="so_arm101_moveit_config"
    ).to_moveit_configs()

    # Load ExecuteTaskSolutionCapability so MoveIt Task Constructor can execute
    # planned solutions. Harmless no-op when MTC isn't being used.
    move_group_capabilities = {
        "capabilities": "move_group/ExecuteTaskSolutionCapability"
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), move_group_capabilities],
    )

    return LaunchDescription([move_group_node])
