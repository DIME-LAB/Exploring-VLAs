#!/usr/bin/env python3
# Reference: Phase 6 verification script for SO-ARM101 collision-free drop motion
"""Verify drop motion doesn't knock cups in Isaac Sim.

Automated verification loop for the SO-ARM101 drop motion pipeline.
Runs the full sequence: grasp_home → drop_point → drop_sweep → check cups →
grasp_home (return) → check cups. Queries Isaac Sim via socket to detect
fallen cups after each motion step.

Requirements:
    - SO-ARM101 control stack running (ros2 launch so_arm101_control control.launch.py)
    - Isaac Sim running with soarm101-dt extension (cups added, drop_poses publishing)
    - Drop targets refreshed in GUI before running

Exit codes:
    0 — all passes succeeded (N consecutive clean passes)
    1 — verification failed (cup knocked or planning error)

Usage:
    python3 verify_drop_collision.py                     # default: drop_1, 3 passes
    python3 verify_drop_collision.py --target drop_0     # test red cup
    python3 verify_drop_collision.py --target drop_2 --passes 5  # 5 passes on blue
    python3 verify_drop_collision.py --passes 1          # quick single-pass check
"""
import argparse
import json
import socket
import subprocess
import sys
import time


def ros2_service(service, timeout=30):
    """Call a Trigger service, return (success, message)."""
    try:
        result = subprocess.run(
            ['ros2', 'service', 'call', service, 'std_srvs/srv/Trigger'],
            capture_output=True, text=True, timeout=timeout)
        output = result.stdout
        if 'success=True' in output:
            return True, output
        elif 'success=False' in output:
            # Extract message
            msg = ''
            if "message='" in output:
                msg = output.split("message='")[1].split("'")[0]
            return False, msg
        return False, output
    except subprocess.TimeoutExpired:
        return False, f'Service call timed out after {timeout}s'


def ros2_param_set(node, param, value):
    """Set a ROS2 parameter."""
    subprocess.run(
        ['ros2', 'param', 'set', node, param, value],
        capture_output=True, text=True, timeout=10)


def check_cups_isaac_sim(port=8767):
    """Query Isaac Sim for cup states. Returns dict of cup_name -> {upright, pos}."""
    code = '''
import omni.usd
from pxr import UsdGeom
stage = omni.usd.get_context().get_stage()
lines = []
for cup_name in ['cup_red', 'cup_green', 'cup_blue']:
    prim = stage.GetPrimAtPath(f'/World/Containers/{cup_name}')
    if not prim:
        lines.append(f'{cup_name}|NOT_FOUND|0|0|0')
        continue
    xf = UsdGeom.Xformable(prim)
    world_tf = xf.ComputeLocalToWorldTransform(0)
    pos = world_tf.ExtractTranslation()
    z_axis = world_tf.GetRow3(2)
    upright = 'True' if abs(z_axis[2]) > 0.9 else 'False'
    lines.append(f'{cup_name}|{upright}|{pos[0]*1000:.1f}|{pos[1]*1000:.1f}|{pos[2]*1000:.1f}')
result = ';'.join(lines)
'''
    s = socket.socket()
    s.settimeout(15)
    s.connect(('localhost', port))
    s.sendall(json.dumps({'type': 'execute_python_code', 'params': {'code': code}}).encode())
    d = b''
    while True:
        c = s.recv(16384)
        if not c:
            break
        d += c
        try:
            json.loads(d.decode())
            break
        except json.JSONDecodeError:
            continue
    s.close()
    resp = json.loads(d.decode())
    raw = resp.get('result', '')
    if isinstance(raw, dict):
        raw = raw.get('result', str(raw))

    cups = {}
    for entry in raw.split(';'):
        parts = entry.split('|')
        if len(parts) == 5:
            name, upright, x, y, z = parts
            cups[name] = {
                'upright': upright == 'True',
                'pos': (float(x), float(y), float(z)),
            }
    return cups


def cups_all_upright(cups):
    """Check all cups are upright."""
    for name, info in cups.items():
        if not info['upright']:
            return False, name
    return True, None


def run_step(name, service, timeout=30):
    """Run a service step, print result."""
    print(f'  {name}...', end=' ', flush=True)
    ok, msg = ros2_service(service, timeout=timeout)
    status = 'OK' if ok else f'FAIL ({msg})'
    print(status)
    return ok


def run_pass(target, pass_num):
    """Run one full pass. Returns (success, failed_step)."""
    print(f'\n=== Pass {pass_num} — target: {target} ===')

    # Check cups before starting
    cups_before = check_cups_isaac_sim()
    ok, fallen = cups_all_upright(cups_before)
    if not ok:
        print(f'  !! Cup {fallen} already fallen before pass — SKIP')
        return False, 'pre-check'
    print(f'  Pre-check: all cups upright')

    # Step 1: Home
    if not run_step('Grasp Home', '/so_arm101_control_gui/grasp_home', timeout=60):
        return False, 'grasp_home'
    time.sleep(1)

    # Step 2: Select target
    ros2_param_set('/so_arm101_control_gui', 'ik_target', target)
    if not run_step('Drop Select', '/so_arm101_control_gui/drop_select'):
        return False, 'drop_select'

    # Step 3: Drop Point
    if not run_step('Drop Point', '/so_arm101_control_gui/drop_point'):
        return False, 'drop_point'
    time.sleep(1)

    # Step 4: Drop Sweep
    if not run_step('Drop Sweep', '/so_arm101_control_gui/drop_sweep', timeout=60):
        return False, 'drop_sweep'
    time.sleep(2)

    # Check cups after sweep
    cups_after_sweep = check_cups_isaac_sim()
    ok, fallen = cups_all_upright(cups_after_sweep)
    if not ok:
        print(f'  !! Cup {fallen} knocked over AFTER SWEEP')
        # Report which cup and position change
        before_pos = cups_before[fallen]['pos']
        after_pos = cups_after_sweep[fallen]['pos']
        print(f'     Before: {before_pos}')
        print(f'     After:  {after_pos}')
        return False, f'sweep_collision_{fallen}'
    print(f'  Post-sweep check: all cups upright ✓')

    # Step 5: Home again
    if not run_step('Grasp Home (return)', '/so_arm101_control_gui/grasp_home', timeout=60):
        return False, 'grasp_home_return'
    time.sleep(2)

    # Check cups after home
    cups_after_home = check_cups_isaac_sim()
    ok, fallen = cups_all_upright(cups_after_home)
    if not ok:
        print(f'  !! Cup {fallen} knocked over AFTER HOME RETURN')
        before_pos = cups_after_sweep[fallen]['pos']
        after_pos = cups_after_home[fallen]['pos']
        print(f'     Before: {before_pos}')
        print(f'     After:  {after_pos}')
        return False, f'home_collision_{fallen}'
    print(f'  Post-home check: all cups upright ✓')

    print(f'  === Pass {pass_num} PASSED ===')
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='drop_1', help='Drop target ID')
    parser.add_argument('--passes', type=int, default=3, help='Required consecutive passes')
    args = parser.parse_args()

    print(f'Drop collision verification: {args.passes} passes on {args.target}')
    print(f'=' * 60)

    # Refresh drop targets first
    print('\nRefreshing drop targets...')
    ok, _ = ros2_service('/so_arm101_control_gui/drop_refresh')
    if not ok:
        # Retry after delay
        time.sleep(3)
        ok, _ = ros2_service('/so_arm101_control_gui/drop_refresh')
    time.sleep(2)

    consecutive = 0
    total_attempts = 0
    failures = {}

    while consecutive < args.passes and total_attempts < args.passes * 3:
        total_attempts += 1
        passed, failed_step = run_pass(args.target, total_attempts)

        if passed:
            consecutive += 1
            print(f'\n  Consecutive passes: {consecutive}/{args.passes}')
        else:
            consecutive = 0
            failures[failed_step] = failures.get(failed_step, 0) + 1
            print(f'\n  Reset to 0 consecutive passes')
            # Don't continue if cups are fallen — need manual reset
            if 'collision' in (failed_step or ''):
                print(f'\n  Cup knocked over — stopping. Reset cups and rerun.')
                break

    print(f'\n{"=" * 60}')
    if consecutive >= args.passes:
        print(f'VERIFICATION PASSED: {args.passes} consecutive clean passes')
    else:
        print(f'VERIFICATION FAILED after {total_attempts} attempts')
        print(f'Failure breakdown:')
        for step, count in sorted(failures.items(), key=lambda x: -x[1]):
            print(f'  {step}: {count} failures')

    return 0 if consecutive >= args.passes else 1


if __name__ == '__main__':
    sys.exit(main())
