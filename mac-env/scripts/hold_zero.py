#!/usr/bin/env python3
"""Drive to 0,0,0,0,0,0 and hold there for 30 seconds — long enough for
user to physically inspect the pose against URDF-zero expectations.

Publishes target to /joint_targets so gravity_comp_node applies FF +
integrator, putting actual position at target (within FF/integrator
residual ~0.3°). User can then verify visually that the resulting pose
matches the URDF-zero geometric reference.
"""
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper_joint']
TARGET = {n: 0.0 for n in JOINT_NAMES}
HOLD_S = 30.0

def quintic(a):
    return 10*a**3 - 15*a**4 + 6*a**5

def main():
    rclpy.init()
    n = Node('hold_zero')
    pub = n.create_publisher(JointState, '/joint_targets', 50)
    actual = {}
    n.create_subscription(JointState, '/joint_states',
                          lambda m: actual.update(zip(m.name, m.position)), 50)
    deadline = time.time() + 5.0
    while time.time() < deadline and not actual:
        rclpy.spin_once(n, timeout_sec=0.1)
    if not actual:
        print("FAIL: no /joint_states")
        return 1

    start = {j: actual.get(j, 0.0) for j in JOINT_NAMES}
    print(f"Starting pose: { {k: round(math.degrees(v),1) for k,v in start.items()} }")

    # 4s smooth move to zero
    rate = 50
    n_steps = int(4.0 * rate)
    print("smooth-moving → 0,0,0,0,0,0 (4s)")
    for i in range(n_steps + 1):
        a = i / n_steps
        s = quintic(a)
        interp = {j: start[j] + s * (TARGET[j] - start[j]) for j in JOINT_NAMES}
        msg = JointState(); msg.header.stamp = n.get_clock().now().to_msg()
        msg.name = list(interp.keys()); msg.position = list(interp.values())
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.0)
        time.sleep(1.0 / rate)

    # Hold for HOLD_S, printing measured pose every 5s
    print(f"\nHOLDING at zero for {HOLD_S}s — please inspect the arm physically.")
    print("URDF-zero reference: arm pointing FORWARD, upper arm HORIZONTAL,")
    print("forearm continuing in same line, wrist NOT bent, gripper horizontal.")
    print()
    t0 = time.time()
    last_report = 0
    while time.time() - t0 < HOLD_S:
        msg = JointState(); msg.header.stamp = n.get_clock().now().to_msg()
        msg.name = list(TARGET.keys()); msg.position = list(TARGET.values())
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.0)
        elapsed = int(time.time() - t0)
        if elapsed >= last_report + 5:
            last_report = elapsed
            cur = {j: round(math.degrees(actual.get(j, 0)), 2) for j in JOINT_NAMES}
            print(f"  t={elapsed:>2}s  actual: {cur}°")
        time.sleep(1.0 / rate)

    cur = {j: round(math.degrees(actual.get(j, 0)), 2) for j in JOINT_NAMES}
    print(f"\nFINAL actual at target=0,0,0,0,0,0: {cur}°")
    n.destroy_node(); rclpy.shutdown()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
