#!/bin/bash
# _param_set_string.sh — set a string-typed ROS2 param without int-coercion.
#
# Why this exists: `ros2 param set <node> <name> '4'` is parsed by the CLI as
# integer 4, which fails when the parameter is declared as Type.STRING (the
# control_gui's widget_value param is one such case). Quoting doesn't help.
# This helper goes through rcl_interfaces/SetParameters with explicit string typing.
#
# Usage:
#   _param_set_string.sh <param_name> <value>            # node defaults to /so_arm101_control_gui
#   _param_set_string.sh <param_name> <value> <node>
#
# Examples:
#   _param_set_string.sh widget_id Episodes
#   _param_set_string.sh widget_value 4
#   _param_set_string.sh ik_target red_2x3_0

set -eu

NAME="${1:?param name required}"
VALUE="${2:?param value required}"
NODE="${3:-/so_arm101_control_gui}"

python3 - "$NODE" "$NAME" "$VALUE" <<'PY'
import sys
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

node_name, param_name, param_value = sys.argv[1], sys.argv[2], sys.argv[3]

rclpy.init()
n = Node('param_set_string')
cli = n.create_client(SetParameters, f'{node_name}/set_parameters')
if not cli.wait_for_service(timeout_sec=3.0):
    print(f'ERROR: {node_name}/set_parameters not available', file=sys.stderr)
    sys.exit(1)

req = SetParameters.Request()
req.parameters = [Parameter(
    name=param_name,
    value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=param_value),
)]
fut = cli.call_async(req)
rclpy.spin_until_future_complete(n, fut, timeout_sec=5.0)
res = fut.result()
n.destroy_node()
rclpy.shutdown()

if res is None or not res.results:
    print(f'ERROR: no response setting {param_name}={param_value}', file=sys.stderr)
    sys.exit(1)
r = res.results[0]
if not r.successful:
    print(f'ERROR: set failed: {r.reason}', file=sys.stderr)
    sys.exit(1)
print(f'OK: {node_name} {param_name}={param_value!r}')
PY
