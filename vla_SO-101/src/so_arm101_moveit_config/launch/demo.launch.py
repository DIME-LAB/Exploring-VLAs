from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "so_arm101", package_name="so_arm101_moveit_config"
    ).to_moveit_configs()
    ld = generate_demo_launch(moveit_config)

    # Add end-effector pose publisher (TF lookup: base -> gripper)
    ee_pose_node = Node(
        package="so_arm101_control",
        executable="ee_pose_publisher",
        name="ee_pose_publisher",
        output="screen",
    )
    ld.add_action(ee_pose_node)

    # Add control GUI
    gui_node = Node(
        package="so_arm101_control",
        executable="control_gui",
        name="so_arm101_control_gui",
        output="screen",
    )
    ld.add_action(gui_node)

    return ld
