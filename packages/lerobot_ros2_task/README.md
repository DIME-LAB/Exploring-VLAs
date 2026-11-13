# RunTask policy episodes (2.5) and prompt/model controls (2.5b)

This is an opt-in ROS action server and pip package. It uses the standalone robot
and camera plugins. Build [lerobot_ros2_interfaces](../lerobot_ros2_interfaces/README.md)
with your ROS distribution and Python version, then install the context, camera,
robot and task distributions in an isolated ROS-aware Python 3.12 environment.
`rclpy`, control_msgs, trajectory_msgs, sensor_msgs, std_msgs and the generated
RunTask interface come from ROS, not PyPI. LeRobot policy dependencies (for example
`lerobot[smolvla]`) must be installed for the model you select.

On dual-a4500 the existing deployment environment is left alone. A venv made from
its pixi Python with `--system-site-packages` can use the installed dependencies;
install these packages only into that new venv. Run its interpreter through
`pixi run --manifest-path linux-env/pixi.toml` for the ROS native library paths.
Build the action with that same pixi environment and Python (Jazzy/Python 3.12).
A Humble/Python 3.10 generated overlay cannot be used by the Jazzy interpreter.
The wheel also installs the SO-101 example under `sys.prefix/share/lerobot_ros2_task/examples/so101.json`;
use that path with `--robot-config` when working outside a repository checkout.

```bash
# From the repository root, after building/sourcing the action interface:
source "$HOME/w_phase2-interfaces/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/linux-env/cyclonedds.xml"
export ROS_DOMAIN_ID=0  # standard deployment; live use needs the lead's sim window
# Resolve/pin GPU 1 as deployment does. Supply its UUID, not CUDA's device order.
export CUDA_VISIBLE_DEVICES='<GPU-1 UUID from nvidia-smi>'

lerobot-ros2-run-task \
  --robot-config packages/lerobot_ros2_task/examples/so101.json \
  --models-file /path/to/models.json --ros-args \
  -p model:=/path/to/pretrained_model \
  -p fps:=17.4 -p device:=cuda
```

The models file is a JSON object such as `{"base-ft":"/path/to/pretrained_model"}`.
The file options set string parameters directly, avoiding ROS YAML command-line
interpretation of nested JSON. They can also be supplied as properly quoted string
parameters in a ROS parameter YAML file.

The example retains `/joint_states` for old models. Models trained on physical
feedback require `/joint_states_measured` in robot_config; do not change only the
model path when the state semantics differ. Empty robot_config uses the SO-101
preset without cameras; a vision policy requires camera configs, as in the example.
Generic robot configs are accepted too, with matching home positions. The default
home matches the deployed evaluation home, in degrees. Override the JSON `home`
parameter if joint names or units differ. `home_duration_s` defaults to 3;
`home_timeout_s` defaults to 15. `fps` also sets streaming waypoint duration to
1/fps. Only one normal task goal may be active at a time.

```bash
lerobot-ros2-task-client --instruction 'Pick a blue lego and place it in blue cup' \
  --max-duration 60 --model base-ft --wait
lerobot-ros2-task-client --cancel
```

RunTask action goals have `instruction`, positive finite `max_duration` in wall
seconds, optional `model`, and `update_instruction` (false by default). A normal
goal goes through home → run → policy stop → return home. Success, timeout, cancel
and errors all attempt the final home. Result success is true only for a success
signal followed by a completed return-home move. FJT home goals check both action
status and controller result. A failed return home reports error. Cancel can
interrupt initial homing; final homing is allowed to finish. max_duration covers
the run phase, including model startup; it does not cut short return home.

A separate task-specific scorer publishes `std_msgs/Bool(data=true)` on
`/run_task/success` while the policy is running. The node clears success at goal
start and ignores signals outside the run phase. This is an input contract, not
an automatic claim that an object reached a cup; connect the intended evaluator.
Feedback reports `home`, `run`, `return_home`, live instruction/model and elapsed
run time. Timeout and error abort the action; action cancellation reports canceled
only after final home. Result transitions supply the completed response on rclpy versions that publish
it immediately, avoiding an empty result for clients waiting before completion.
The node also supports the callback-return API used by other ROS releases.
Process termination is not the action-cancel interface;
use the cancel client and wait for result before stopping ROS.

Policy loading, fresh image/state capture, preprocessing and chunk inference run
in a spawned process. The action loop never captures images or waits for a policy
chunk. It uses absolute monotonic deadlines, skips missed slots and discards
expired chunk prefixes instead of replaying a burst. Streaming uses the same
positions-only depth-one controller topics as the fork. Model outputs pass through
LeRobot's saved pre/postprocessor pipelines. There is no RTC guidance in this
runtime; ROADMAP 2.8 retains that work, and the deployed fork dispatcher/server
continue to own the current RTC path. This node has not replaced deployment or
established Phase 1 evaluation parity. It uses ordinary thread scheduling; the
Phase 1 FIFO10 dispatcher is still in the deployed fork.

## Prompt changes and model selection

```bash
# Both enter the same action server using update_instruction=true.
lerobot-ros2-task-client --update-instruction --instruction 'Place the blue lego in the blue cup'
ros2 topic pub --once /run_task/instruction std_msgs/msg/String \
  '{data: "Place the blue lego in the blue cup"}'
```

An update-only goal requires a nonempty instruction, an active run and an empty
model. It updates the current episode without homing. Shared-memory instruction
updates discard buffered old-prompt actions and tag new chunks with a generation;
a chunk already being computed under an old instruction cannot enter the new
queue. No new action is sent until a current-instruction chunk is ready.
Model selection belongs to a normal goal. Cancel/wait for the active goal before
starting another model; loading never swaps weights during an active trajectory.
Topic updates are translated into update-only RunTask goals, and the panel uses
the same client/action path.

The pip wheel includes `soarm101_task_tools`, an optional bridge tool module and
panel. Load it only during a granted sim window using an opt-in launch configuration:

```bash
# In the environment used to start Kit (do not edit the deployed launcher):
export KIT_MCP_TOOL_PATHS='/path/to/Exploring-VLAs/isaac:/path/to/installed/task/site-packages'
export KIT_MCP_TOOL_MODULES='soarm101_tools,soarm101_task_tools'
export LEROBOT_RUN_TASK_COMMAND='["/absolute/path/to/sourced-task-client-wrapper"]'
```

The wrapper must source the generated ROS overlay, activate the intended ROS
libraries, set the explicit CycloneDDS URI and invoke
`exec lerobot-ros2-task-client "$@"` in the task venv. It runs
outside Isaac's Python. The bridge calls that JSON argv directly, with no shell
expansion. The module owns the distinct `lerobot_task` namespace. Attach the
requested `soarm101.run_task`, `soarm101.set_task_instruction`, and
`soarm101.cancel_task` aliases explicitly to the existing SO-ARM tool owner;
the bridge permits only one owner per namespace. To attach aliases and show the
panel inside Kit with the module on its Python path, call through `run_python`:

```python
from omni_kit_mcp import get_mcp_bridge
from soarm101_task_tools import attach_to_soarm101
from soarm101_task_tools.panel import show
attach_to_soarm101(get_mcp_bridge())
show()
```

Alias attachment can be repeated. Reattach after reloading `soarm101_tools`, which
removes its aliases. The owner must permit additions; a sealed owner refuses them.

The panel has instruction/model fields, maximum duration, Run task, Update prompt
and Cancel and home. It does not change the deployed `isaac/exts/` extension or
auto-open on current launches. Its UI result is goal acceptance, not task success;
use action feedback/result to observe episode completion.

## Verification

`python -m pytest packages/tests -q` runs fake-message tests against real LeRobot
interfaces/factories; fork comparisons run when that source is installed. After
building and sourcing the interface overlay, run native checks in an isolated
domain assigned by the lead, with the same transport config:

```bash
ROS_DOMAIN_ID=93 RUN_NATIVE_ROS=1 python -m pytest \
  packages/tests/test_native_lifecycle.py packages/tests/test_native_task.py -q
```

Those checks use actual Jazzy action/messages/executors with fake robot/model
owners. Sim/controller/model checks require a separate window and are not replaced
by these tests. The workflow in `.github/workflows/ros2-plugins.yml` builds wheels,
checks PyPI and fork interfaces, builds the action and runs native transport checks.

For CycloneDDS an explicit `CYCLONEDDS_URI` is required, including attachment to
an initialized context. The node, client and context module refuse
to use `rmw_cyclonedds_cpp` when it is absent and name the project's
`linux-env/cyclonedds.xml` loopback config. Bare installed Jazzy/Cyclone node
teardown can hang when the transport config is unset.

Apache-2.0, Copyright 2026 Exploring-VLAs contributors. ROS interaction uses
standard rclpy and generated interfaces. Policy loading uses Hugging Face LeRobot's
public policy/preprocessor interfaces; no other project's RunTask implementation
is included. See LICENSE.
