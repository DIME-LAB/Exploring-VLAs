"""Small ROS stand-ins used only by offline tests."""
import sys
import importlib.machinery
import os
import threading
import types
from concurrent.futures import Future


class Context:
    def __init__(self):
        self.running = False
        self.starts = self.stops = 0

    def ok(self):
        return self.running

    def shutdown(self):
        self.running = False
        self.stops += 1


class Executor:
    def __init__(self, context):
        self.context = context
        self.nodes = []
        self.stops = 0
        self.event = threading.Event()

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)

    def spin_once(self, timeout_sec):
        self.event.wait(timeout_sec)
        self.event.clear()

    def wake(self):
        self.event.set()

    def shutdown(self):
        self.stops += 1


class Node:
    def __init__(self, name, context=None):
        self.name = name
        self.context = context
        self.subs = []
        self.pubs = []
        self.destroyed = False
        self.params = {}

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return Message(value=self.params[name])

    def get_logger(self):
        import logging
        return logging.getLogger(self.name)

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = types.SimpleNamespace(msg_type=msg_type, topic=topic, callback=callback, qos=qos)
        self.subs.append(sub)
        return sub

    def destroy_subscription(self, sub):
        self.subs.remove(sub)

    def create_publisher(self, msg_type, topic, qos):
        pub = types.SimpleNamespace(msg_type=msg_type, topic=topic, qos=qos, messages=[])
        pub.publish = pub.messages.append
        self.pubs.append(pub)
        return pub

    def destroy_publisher(self, pub):
        self.pubs.remove(pub)

    def destroy_node(self):
        self.destroyed = True


class Message:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class JointTrajectory(Message):
    def __init__(self, **kw):
        self.joint_names = []
        self.points = []
        self.header = Message(stamp=Message(sec=0, nanosec=0))
        super().__init__(**kw)


class FollowJointTrajectory:
    class Goal:
        def __init__(self):
            self.trajectory = JointTrajectory()


def completed(value):
    future = Future()
    future.set_result(value)
    return future


class GoalHandle:
    accepted = True

    def __init__(self):
        self.result = Future()
        self.cancels = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancels += 1
        return completed(Message(goals_canceling=[self]))


class ActionClient:
    def __init__(self, node, action_type, topic, **kwargs):
        self.topic = topic
        self.goals = []
        self.handles = []
        self.destroyed = False

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        handle = GoalHandle()
        self.handles.append(handle)
        return completed(handle)

    def destroy(self):
        self.destroyed = True


class ActionServer:
    def __init__(self, *args, **kwargs):
        self.callbacks = kwargs

    def destroy(self):
        pass


def install(monkeypatch):
    context = Context()
    default = [context]

    def module(name, **kw):
        obj = types.ModuleType(name)
        obj.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        obj.__path__ = []
        obj.__dict__.update(kw)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    def init(*, context=None):
        context = context or default[0]
        if context.starts:
            raise RuntimeError("Context.init() must only be called once")
        context.starts += 1
        context.running = True

    def shutdown():
        if default[0].ok():
            default[0].shutdown()
        default[0] = Context()

    ros = module("rclpy", init=init, get_default_context=lambda: default[0],
                 get_rmw_implementation_identifier=lambda: os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
                 create_node=lambda name, context=context: Node(name, context),
                 ok=lambda: default[0].ok(), shutdown=shutdown, try_shutdown=shutdown)
    ros.node = module("rclpy.node", Node=Node)
    ros.subscription = module("rclpy.subscription", Subscription=Message)
    ros.executors = module("rclpy.executors", SingleThreadedExecutor=Executor)
    ros.action = module("rclpy.action", ActionClient=ActionClient, ActionServer=ActionServer,
                        GoalResponse=Message(ACCEPT=1, REJECT=0), CancelResponse=Message(ACCEPT=1))
    module("rclpy.callback_groups", ReentrantCallbackGroup=Message)
    module("std_msgs")
    module("std_msgs.msg", String=Message, Bool=Message)
    policies = Message(BEST_EFFORT=1, RELIABLE=2, VOLATILE=1, TRANSIENT_LOCAL=2, KEEP_LAST=1)
    ros.qos = module("rclpy.qos", QoSProfile=Message, ReliabilityPolicy=policies,
                     DurabilityPolicy=policies, HistoryPolicy=policies,
                     qos_profile_sensor_data=Message(depth=5))
    for name, classes in {
        "sensor_msgs.msg": {"Image": Message, "CompressedImage": Message, "JointState": Message},
        "trajectory_msgs.msg": {"JointTrajectory": JointTrajectory, "JointTrajectoryPoint": Message},
        "builtin_interfaces.msg": {"Duration": Message},
        "control_msgs.action": {"FollowJointTrajectory": FollowJointTrajectory},
    }.items():
        module(name.split('.')[0])
        module(name, **classes)
    return ros, context
