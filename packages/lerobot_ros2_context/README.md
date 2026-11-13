# Shared ROS 2 context (2.1)

`pip install ./packages/lerobot_ros2_context` in an environment that supplies
`rclpy`. ROS dependencies come from ROS/RoboStack, not PyPI.

```python
from lerobot_ros2_context import acquire
lease = acquire()  # one node/executor shared by process-local users
node = lease.node
lease.close()      # release after destroying this user's subscriptions/publishers
```

`acquire(context=my_context, executor=my_executor)` attaches a plugin node to
an existing executor. Its owner must spin it. The module never stops an external
executor and shuts down a context only when it initialized that context itself.
An initialized context without an executor gets a managed executor. Last release
stops the managed spin thread before destroying its node. Repeated close is OK.
A callback failure is available through `lease.check()` and subsequent acquire.

Apache-2.0. Copyright 2026 Exploring-VLAs contributors. Lifecycle is implemented
here independently; camera/robot lineage is documented by their distributions.

For CycloneDDS, set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and an explicit
`CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml`
(loopback transport). The context module refuses to initialize CycloneDDS with
that setting absent. The bare installed Jazzy/Cyclone node teardown can hang
without this explicit transport config. Other projects may supply their own XML.
