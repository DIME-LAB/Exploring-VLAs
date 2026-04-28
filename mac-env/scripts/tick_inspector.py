#!/usr/bin/env python3
"""Inspect /joint_commands and /joint_states timeline to find discrete
'ticks' — places where one signal moves in step-changes large enough
for a human eye to see a discrete event.

Approach: for each topic and each joint, compute the per-message position
delta. Find samples where |delta| exceeds the encoder LSB (0.088°).
Cluster nearby exceedances into events. Print a timeline.
"""
import json
import math
import sys
from collections import defaultdict


LSB_DEG = 360.0 / 4096.0  # 0.088° per encoder count


def load(path):
    cmds = []
    states = []
    for line in open(path):
        r = json.loads(line)
        if r['kind'] == 'c':
            cmds.append(r)
        else:
            states.append(r)
    cmds.sort(key=lambda r: r['t'])
    states.sort(key=lambda r: r['t'])
    if cmds:
        t0 = cmds[0]['t']
        for r in cmds:
            r['t'] -= t0
        for r in states:
            r['t'] -= t0
    return cmds, states


def per_joint_series(rows):
    """Returns dict joint_name -> list of (t, pos_deg)."""
    out = defaultdict(list)
    for r in rows:
        for n, p in zip(r['name'], r['pos']):
            out[n].append((r['t'], math.degrees(p)))
    return out


def find_steps(series, threshold_deg):
    """Find consecutive samples where |Δpos| > threshold."""
    events = []
    for i in range(1, len(series)):
        t_prev, p_prev = series[i - 1]
        t_now, p_now = series[i]
        dp = p_now - p_prev
        if abs(dp) >= threshold_deg:
            events.append({
                't': t_now,
                'dt_ms': (t_now - t_prev) * 1000,
                'dp_deg': dp,
                'p_prev': p_prev,
                'p_now': p_now,
            })
    return events


def cluster_events(events, gap_s=0.05):
    """Group events within gap_s of each other."""
    clusters = []
    cur = []
    for e in events:
        if cur and (e['t'] - cur[-1]['t']) > gap_s:
            clusters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        clusters.append(cur)
    return clusters


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/tick_capture.jsonl'
    cmds, states = load(path)
    print(f"# {len(cmds)} cmds, {len(states)} states from {path}")
    cmd_by_joint = per_joint_series(cmds)
    state_by_joint = per_joint_series(states)

    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll']

    # For each joint, find LSB-crossing events on /joint_states (= visible
    # arm motion) and on /joint_commands (= what we sent). Threshold a
    # bit above LSB to filter measurement noise.
    print("\n## State (servo physical) — events ≥ 1 LSB (0.088°) per sample:\n")
    print("| joint | event_count | first 5 events: t_s, dt_ms, dp_deg |")
    print("|---|---:|---|")
    for j in joints:
        events = find_steps(state_by_joint.get(j, []), LSB_DEG)
        sample = events[:5]
        s = '; '.join(
            f"t={e['t']:.2f} dt={e['dt_ms']:.0f}ms dp={e['dp_deg']:+.2f}°"
            for e in sample)
        print(f"| {j} | {len(events)} | {s} |")

    # Look at /joint_commands deltas per joint
    print("\n## Cmd wire — events ≥ 0.05° per sample (smaller threshold to see r_hat ramps):\n")
    print("| joint | event_count | mean_dp_deg | max_dp_deg |")
    print("|---|---:|---:|---:|")
    for j in joints:
        events = find_steps(cmd_by_joint.get(j, []), 0.05)
        if events:
            dps = [abs(e['dp_deg']) for e in events]
            print(f"| {j} | {len(events)} | {sum(dps)/len(dps):.3f} | {max(dps):.3f} |")
        else:
            print(f"| {j} | 0 | - | - |")

    # Detect "post-motion ticks": after each /joint_commands gap >0.5s
    # (would indicate idle), look for state-side events.
    # Actually with continuous publishing there are no gaps, so use a
    # different detector: identify motion segments via cmd-velocity.
    print("\n## Per-joint state-side discrete jumps over time, focusing on a single joint:\n")
    # Pick shoulder_lift (most-loaded) and show every state event with dt
    j = 'shoulder_lift'
    events = find_steps(state_by_joint.get(j, []), LSB_DEG)
    print(f"### {j} — {len(events)} discrete servo movements")
    print()
    print("| t (s) | dt (ms) | step (deg) | running pos (deg) |")
    print("|---:|---:|---:|---:|")
    cum = 0.0
    last_t = -100
    for e in events[:80]:
        # Show roughly every event but compress dense bursts
        cum = e['p_now']
        print(f"| {e['t']:.2f} | {e['dt_ms']:.0f} | {e['dp_deg']:+.3f} | {cum:+.2f} |")
        last_t = e['t']

    # Also inspect cmd-side what's happening at the same timestamps
    print(f"\n### {j} cmd timeline at those same timestamps:\n")
    print("| t (s) | cmd_pos | cmd_velocity_deg_s |")
    print("|---:|---:|---:|")
    cmd_series = cmd_by_joint.get(j, [])
    if cmd_series:
        for e in events[:20]:
            t = e['t']
            # Find nearest cmd before and after
            idx = next((i for i, (tc, _) in enumerate(cmd_series) if tc >= t), len(cmd_series) - 1)
            if 0 < idx < len(cmd_series) - 1:
                t_a, p_a = cmd_series[idx - 1]
                t_b, p_b = cmd_series[idx + 1]
                dt_cmd = t_b - t_a
                vel = (p_b - p_a) / dt_cmd if dt_cmd > 0 else 0.0
                print(f"| {t:.2f} | {p_a:+.2f} | {vel:+.2f} |")


if __name__ == '__main__':
    main()
