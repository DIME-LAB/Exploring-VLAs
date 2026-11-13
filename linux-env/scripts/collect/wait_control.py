"""Wait for this instance's three controllers using services, without the ROS daemon."""
import json
import os
from pathlib import Path
import time
import rclpy
from controller_manager_msgs.srv import ListControllers
from storage import prepare_environment


def main():
    runtime = prepare_environment(int(os.environ.get('SOARM_INSTANCE_INDEX', '0')))
    rclpy.init()
    node = rclpy.create_node('collection_control_ready')
    client = node.create_client(ListControllers, '/controller_manager/list_controllers')
    required = {'arm_controller', 'gripper_controller', 'joint_state_broadcaster'}
    deadline = time.monotonic() + 60
    states = {}
    try:
        while time.monotonic() < deadline:
            if not client.wait_for_service(timeout_sec=1.):
                continue
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.)
            if future.done() and future.result():
                states = {c.name:c.state for c in future.result().controller}
                if all(states.get(name) == 'active' for name in required):
                    (runtime / 'control-ready.json').write_text(json.dumps(dict(ready=True,controllers=states),indent=2))
                    print('Control ready: arm, gripper and state broadcaster active',flush=True)
                    return
            rclpy.spin_once(node, timeout_sec=.1)
        raise TimeoutError(f'Instance controllers did not become active: {states}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
