# ROS resource retirement for collection

The frozen camera smoke stopped before episodes: both clock trajectories passed,
then clock_sanity exited with SIGSEGV. A GDB rerun located the failure in Cyclone's
recvUC thread while the main thread dlclose'd librmw_implementation during exit;
DDS gc/receive threads remained alive. The image adapter was not loaded by this
process, and Kit ran in a different process.

A ROS-only actual PickPlaceNode probe isolated the retained owners. After executor
retirement and Node.destroy_node plus rclpy.shutdown, DDS threads remained. Even
explicit context.destroy did not release them. Destroying the retained waitables
immediately removed all DDS threads. Humble's Node.destroy_node handles ordinary
publishers/subscriptions and QoS events, but omits its ActionClient waitables.
The shared node now destroys those ActionClients before calling its base destroy;
GUI scene dispatch/motion behaviour is unchanged.

Collector/clock shutdown requires affirmative executor shutdown, joins the spin
thread, cancels queued pool work and joins callback workers, then destroys the
node and explicitly shuts down/destroys the context. A failed owner retirement
prevents native node/context destruction and fails finalization. Humble-specific
pool handling is centralized in ros_lifecycle.py: its public executor shutdown
only tracks active handlers, while queued handlers captured the earlier shutdown
flag. The real regression constructs and destroys the actual shared node ten
times, checking that its two action clients, waitables and DDS threads retire.
The actual Collector boot probe uses separate contexts for collector and fake
transports so it can exercise real collector context destruction.

Clock checks publish an invocation-bound finalization certificate only after
cleanup, using the collector's certificate protocol and a hash of the clock
result. A passed trajectory flag or zero process exit alone is insufficient.
A late abnormal clock-producer exit still fails the runner even if a certificate
exists. Python faulthandler is enabled in the smoke environment.

Sources: installed Humble rclpy/node.py destroy_node and executors.py shutdown;
[Humble native node destruction](https://github.com/ros2/rclpy/blob/humble/rclpy/src/rclpy/node.cpp#L352-L356),
[context finalization](https://github.com/ros2/rclpy/blob/humble/rclpy/src/rclpy/context.cpp#L53-L84).
The context's native shared ownership is finalized when all dependent owners
release it; shutdown alone only invalidates the ROS context.
