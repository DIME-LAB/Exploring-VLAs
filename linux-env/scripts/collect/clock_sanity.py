#!/usr/bin/env python3
"""Check sim-time FJT execution with physical feedback at slow and uncapped rates."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'vla_SO-ARM101/src/so_arm101_control'))
sys.path.insert(0, str(REPO / 'omni-kit-mcp'))
from kit_mcp.client import call
from so_arm101_control.pick_place import PickPlaceNode, ARM_JOINT_NAMES
from storage import confined


def kit(code):
    return call('run_python', {'code': code}, port=int(os.environ['SOARM_KIT_MCP_PORT']), timeout=30)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = confined(args.output)
    expected_domain = str(90 + int(os.environ.get('SOARM_INSTANCE_INDEX', '0')))
    if os.environ.get('ROS_DOMAIN_ID') != expected_domain:
        raise ValueError('Clock sanity requires this instance\'s assigned ROS domain')
    kit('import sys\nsys.path.insert(0,' + repr(str(REPO / 'linux-env/scripts/collect')) + ')\n'
        'import collection_runtime as c\nc.install("deploy")\nc.runtime().timeline.play()\nresult=True')
    rclpy.init(); node = PickPlaceNode(node_name='collection_clock_sanity', use_sim_time=True,
                                     joint_states_topic='/joint_states_measured')
    ex = MultiThreadedExecutor(num_threads=4); ex.add_node(node)
    thread = threading.Thread(target=ex.spin, daemon=True); thread.start()
    results = []
    try:
        deadline = time.monotonic() + 30
        while not node._initial_sync_done:
            if time.monotonic() > deadline:
                raise TimeoutError('No initial ROS state')
            threading.Event().wait(.1)
        for name, cap, direction in [('slow', 10, 1), ('uncapped', 0, -1)]:
            settings = kit('import carb, omni.timeline\n'
                + "s=carb.settings.get_settings()\n"
                + f"s.set_bool('/app/runLoops/main/rateLimitEnabled',{bool(cap)!r})\n"
                + f"s.set_int('/app/runLoops/main/rateLimitFrequency',{cap})\n"
                + 'omni.timeline.get_timeline_interface().play()\nresult=True')
            current = {n: node._actual_positions[n] for n in ARM_JOINT_NAMES}
            current['shoulder_pan'] += direction * .06
            wall_start, sim_start = time.monotonic(), node.get_clock().now().nanoseconds
            ok = node._send_arm_goal(current, duration_s=2., blocking=True)
            wall = time.monotonic() - wall_start
            sim = (node.get_clock().now().nanoseconds - sim_start) / 1e9
            row = dict(regime=name, ok=ok, sim_seconds=sim, wall_seconds=wall, measured_rtf=sim / wall,
                       pan_error_rad=abs(node._actual_positions['shoulder_pan'] - current['shoulder_pan']),
                       measured_positions=dict(node._actual_positions), goal=current,
                       state_source='isaac_articulation')
            if not ok or abs(sim - 2.) > .4 or row['pan_error_rad'] > .02:
                raise RuntimeError(f'Controller does not follow sim time: {row}')
            if name == 'slow' and row['measured_rtf'] >= 1:
                raise RuntimeError(f'Required RTF regime not achieved: {row}')
            results.append(row); print(json.dumps(row), flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=len(results) == 2, regimes=results), indent=2))
        try:
            kit((REPO / 'linux-env/scripts/collect/configure_clock.py').read_text() + '\nresult=configure(30)')
            kit('import collection_runtime as c\nc.runtime().cancel()\nresult=c.install("collect")')
        finally:
            ex.shutdown(timeout_sec=5);  node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
