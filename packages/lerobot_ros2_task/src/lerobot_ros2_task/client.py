# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
import argparse
import json


def main():
    import rclpy
    from rclpy.action import ActionClient
    from action_msgs.srv import CancelGoal
    from lerobot_ros2_interfaces.action import RunTask
    parser = argparse.ArgumentParser(description="Submit, update or cancel a RunTask goal")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument("--model", default="")
    parser.add_argument("--action", default="run_task")
    parser.add_argument("--update-instruction", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = rclpy.create_node("lerobot_task_client")
    try:
        if args.cancel:
            client = node.create_client(CancelGoal, args.action.rstrip("/") + "/_action/cancel_goal")
            if not client.wait_for_service(timeout_sec=5):
                raise TimeoutError("RunTask cancel service is not available")
            future = client.call_async(CancelGoal.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=10)
            if not future.done():
                raise TimeoutError("RunTask cancel request timed out")
            print(json.dumps({"status": "success", "cancelling": len(future.result().goals_canceling)}))
            return
        client = ActionClient(node, RunTask, args.action)
        if not client.wait_for_server(timeout_sec=5):
            raise TimeoutError("RunTask action server is not available")
        goal = RunTask.Goal()
        goal.instruction, goal.max_duration, goal.model = args.instruction, args.max_duration, args.model
        goal.update_instruction = args.update_instruction
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if not future.done():
            raise TimeoutError("RunTask goal response timed out")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("RunTask goal declined (check active goal, instruction, duration and model)")
        if args.wait or args.update_instruction:
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
            result = result_future.result().result
            print(json.dumps({"status": "success", "outcome": result.outcome,
                              "success": result.success, "message": result.message}))
        else:
            print(json.dumps({"status": "success", "outcome": "accepted"}))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
