"""Isaac Sim backend for the SO-ARM101 stack — placeholder.

Isaac Sim is launched externally (see ~/Documents/isaac-sim-mcp/, branch
so-arm101, MCP socket on localhost:8767). It runs as its own process and
publishes the SO-ARM101 topic contract directly:

  * /joint_states            (TaskGraph → ROS2 ArticulationState)
  * /joint_commands          (consumed by Isaac's ArticulationController)
  * /clock                   (sim time)
  * /objects_poses_sim       (TFMessage from action graph)
  * /objects_bbox_sim        (String JSON from action graph)
  * /drop_poses              (TFMessage from action graph, marker frames)
  * /wrist_camera, /top_camera + /camera_info  (Isaac Sim camera sensors)

Because Isaac is a separate process publishing onto the same DDS domain
core.launch.py uses, no ROS nodes need to be launched here. This file
exists so stack.launch.py mode:=isaac has something to include.

Future work that would belong here:
  * MCP-reachability preflight check (TCP probe localhost:8767)
  * Optional re-publisher / topic remap nodes (e.g., if Isaac's topic
    names diverge from the canonical contract, normalize them here)
  * Isaac-specific parameter overrides for control_gui (e.g., scene
    reset action wired to MCP tools)

Usage (typically via stack.launch.py mode:=isaac):
  ros2 launch so_arm101_control backend_isaac.launch.py
"""

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription([])
