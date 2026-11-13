# RunTask ROS interface (2.5)

Build this ROS interface with colcon in the same ROS distribution and Python
version as the RunTask node, then source that build's install/setup.bash. ROS
interfaces need rosidl generation and are distributed as a ROS package; the
runtime/plugins alongside it are pip packages.

```bash
colcon --log-base "$HOME/w_phase2-interfaces/log" build --base-paths packages/lerobot_ros2_interfaces \
  --build-base "$HOME/w_phase2-interfaces/build" \
  --install-base "$HOME/w_phase2-interfaces/install"
source "$HOME/w_phase2-interfaces/install/setup.bash"
```

RunTask goals supply instruction, maximum run duration in wall seconds, and an
optional model key/path. Empty model selects the node's default. Results distinguish
success, timeout, cancel and error; feedback reports home/run/return_home phase,
live instruction/model and elapsed run time. Apache-2.0, Exploring-VLAs contributors.

For CycloneDDS, set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and an explicit
`CYCLONEDDS_URI=file:///absolute/path/to/Exploring-VLAs/linux-env/cyclonedds.xml`
(loopback transport). The context module refuses to initialize CycloneDDS with
that setting absent. The bare installed Jazzy/Cyclone node teardown can hang
without this explicit transport config. Other projects may supply their own XML.
