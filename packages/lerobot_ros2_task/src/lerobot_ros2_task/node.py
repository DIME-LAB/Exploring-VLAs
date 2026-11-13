# Copyright 2026 Exploring-VLAs contributors. Licensed under Apache-2.0.
import json
import threading
from copy import deepcopy

from .episode import Episode, TaskRequest


def create_node(*, robot=None, policy_factory=None):
    """Construct the real ROS node; injectable owners let offline tests use the same entrance."""
    import draccus
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from std_msgs.msg import Bool, String
    from lerobot_ros2_interfaces.action import RunTask
    from lerobot_robot_ros2_joint import ROS2JointRobot, ROS2JointRobotConfig, so101_config
    from lerobot.robots.config import RobotConfig
    from .policy import PolicyProcess
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()

    class RunTaskNode(Node):
        def __init__(self):
            super().__init__("lerobot_run_task")
            self.declare_parameter("robot_config", "{}")
            self.declare_parameter("home", json.dumps({
                "shoulder_pan.pos": 0.874, "shoulder_lift.pos": -14.829, "elbow_flex.pos": 14.829,
                "wrist_flex.pos": 90.0, "wrist_roll.pos": -87.20969653731267, "gripper.pos": 90.0}))
            self.declare_parameter("model", "")
            self.declare_parameter("models", "{}")
            self.declare_parameter("fps", 17.4)
            self.declare_parameter("device", "cuda")
            self.declare_parameter("home_duration_s", 3.0)
            self.declare_parameter("home_timeout_s", 15.0)
            self._lock = threading.RLock()
            self._reserved = False
            self._episode = self._goal = None
            config_values = json.loads(self.get_parameter("robot_config").value)
            config_values.setdefault("type", "so101_ros2_joint")
            config = draccus.decode(RobotConfig, config_values)
            home = json.loads(self.get_parameter("home").value)
            if set(home) != {f"{name}.pos" for name in config.joint_names}:
                raise ValueError("home must contain every configured joint position")
            config.actuate = True
            config.action_duration_s = 1 / self.get_parameter("fps").value
            self._policy_config = deepcopy(config)
            config.cameras = {}  # Camera capture and model work belong to PolicyProcess.
            self.robot = robot or ROS2JointRobot(config)
            self._owns_robot = robot is None
            if self._owns_robot:
                self.robot.connect()
            self._policy_factory = policy_factory or (lambda: PolicyProcess(
                self._policy_config, device=self.get_parameter("device").value))
            self._callbacks = ReentrantCallbackGroup()
            self._server = ActionServer(self, RunTask, "run_task", execute_callback=self._execute,
                                        goal_callback=self._accept, cancel_callback=self._cancel,
                                        callback_group=self._callbacks)
            self._update_client = ActionClient(self, RunTask, "run_task", callback_group=self._callbacks)
            self._instruction_sub = self.create_subscription(String, "run_task/instruction", self._instruction, 10)
            self._success_sub = self.create_subscription(Bool, "run_task/success", self._success, 10)

        def _model(self, requested):
            name = requested or self.get_parameter("model").value
            models = json.loads(self.get_parameter("models").value)
            model = models.get(name, name)
            if not model:
                raise ValueError("Set a model parameter or supply a model in the goal")
            return model

        def _accept(self, request):
            if request.update_instruction:
                with self._lock:
                    valid = (bool(request.instruction.strip()) and not request.model and
                             self._episode is not None and self._episode._running)
                return GoalResponse.ACCEPT if valid else GoalResponse.REJECT
            try:
                TaskRequest(request.instruction, request.max_duration, self._model(request.model))
            except (ValueError, TypeError):
                return GoalResponse.REJECT
            with self._lock:
                if self._reserved:
                    return GoalResponse.REJECT
                self._reserved = True
            return GoalResponse.ACCEPT

        def _cancel(self, goal):
            with self._lock:
                if self._goal is goal and self._episode is not None:
                    self._episode.cancelled.set()
            return CancelResponse.ACCEPT

        def _instruction(self, msg):
            request = RunTask.Goal()
            request.instruction = msg.data
            request.update_instruction = True
            def accepted(future):
                try:
                    if not future.result().accepted:
                        self.get_logger().warning("Instruction update declined: no task is running or prompt is empty")
                except Exception as exc:
                    self.get_logger().warning(f"Instruction update failed: {exc}")
            self._update_client.send_goal_async(request).add_done_callback(accepted)

        def _success(self, msg):
            with self._lock:
                if msg.data and self._episode is not None and self._episode._running:
                    self._episode.succeeded.set()

        def _execute(self, goal):
            result = RunTask.Result()
            if goal.request.update_instruction:
                try:
                    with self._lock:
                        if self._episode is None:
                            raise RuntimeError("Task has already finished")
                        self._episode.set_instruction(goal.request.instruction)
                    result.success, result.outcome = True, "instruction_updated"
                    goal.succeed()
                except Exception as exc:
                    result.outcome, result.message = "error", str(exc)
                    goal.abort()
                return result
            try:
                request = TaskRequest(goal.request.instruction, goal.request.max_duration,
                                      self._model(goal.request.model))
                episode = Episode(self.robot, self._policy_factory(),
                                  json.loads(self.get_parameter("home").value),
                                  fps=self.get_parameter("fps").value,
                                  home_duration_s=self.get_parameter("home_duration_s").value,
                                  home_timeout_s=self.get_parameter("home_timeout_s").value)
                with self._lock:
                    self._episode, self._goal = episode, goal
                    if goal.is_cancel_requested:
                        episode.cancelled.set()
                def feedback(phase, instruction, model, elapsed):
                    message = RunTask.Feedback()
                    message.phase, message.instruction, message.model, message.elapsed = phase, instruction, model, elapsed
                    goal.publish_feedback(message)
                outcome = episode.run(request, feedback)
                result.success, result.outcome, result.message = outcome.success, outcome.outcome, outcome.message
                if outcome.outcome == "cancel":
                    goal.canceled()
                elif outcome.success:
                    goal.succeed()
                else:
                    goal.abort()
            except Exception as exc:
                result.outcome, result.message = "error", str(exc)
                goal.abort()
            finally:
                with self._lock:
                    self._episode = self._goal = None
                    self._reserved = False
            return result

        def destroy_node(self):
            with self._lock:
                if self._episode is not None:
                    self._episode.cancelled.set()
            self._server.destroy()
            self._update_client.destroy()
            if self._owns_robot:
                self.robot.disconnect()
            return super().destroy_node()

    return RunTaskNode()


def main():
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init()
    node = None
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        node = create_node()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        if node is not None and node._episode is not None:
            node._episode.cancelled.set()
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
