#!/usr/bin/env python3
"""For each servo motion event (state delta >= 1 LSB), classify whether
it occurred during active commanded motion or after the trajectory ended.

A 'commanded motion' window is detected by: in the surrounding ±0.5s of
/joint_commands, |Δcmd| >= 0.5° somewhere (= the trajectory was actively
shaping cmd in this neighborhood). Events outside such windows = idle
post-motion ticks.
"""
import json
import math
import sys
from collections import defaultdict


LSB_DEG = 360.0 / 4096.0


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


def per_joint(rows):
    out = defaultdict(list)
    for r in rows:
        for n, p in zip(r['name'], r['pos']):
            out[n].append((r['t'], math.degrees(p)))
    return out


def cmd_motion_at(cmd_series, t, half_window_s=0.5, threshold_deg=0.3):
    """True if cmd was actively moving in [t-window, t+window]."""
    # Find samples in window
    nearby = [p for (tt, p) in cmd_series if abs(tt - t) <= half_window_s]
    if len(nearby) < 2:
        return False
    return (max(nearby) - min(nearby)) >= threshold_deg


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/user_drive_v3.jsonl'
    cmds, states = load(path)
    cmd_by_joint = per_joint(cmds)
    state_by_joint = per_joint(states)

    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll']

    print(f"# {len(cmds)} cmds, {len(states)} states\n")
    print("## Servo movement events classified as ACTIVE-MOTION vs IDLE-DRIFT\n")
    print("| joint | total_events | during_motion | post_motion (= ticks the user sees) |")
    print("|---|---:|---:|---:|")
    summary = {}
    per_joint_classified = {}
    for j in joints:
        s = state_by_joint.get(j, [])
        c = cmd_by_joint.get(j, [])
        events = []
        for i in range(1, len(s)):
            t_prev, p_prev = s[i - 1]
            t_now, p_now = s[i]
            dp = p_now - p_prev
            if abs(dp) >= LSB_DEG:
                in_motion = cmd_motion_at(c, t_now)
                events.append((t_now, dp, in_motion))
        n_during = sum(1 for _, _, m in events if m)
        n_idle = sum(1 for _, _, m in events if not m)
        per_joint_classified[j] = events
        summary[j] = (len(events), n_during, n_idle)
        print(f"| {j} | {len(events)} | {n_during} | {n_idle} |")

    # Show timeline of POST-MOTION (idle-drift) ticks for the joints with the most
    print("\n## POST-MOTION (idle-drift) ticks per joint with timestamps:\n")
    for j in joints:
        events = per_joint_classified[j]
        idle = [(t, dp) for (t, dp, m) in events if not m]
        if not idle:
            continue
        print(f"\n### {j} — {len(idle)} idle-drift ticks")
        print()
        print("| t (s) | step (deg) |")
        print("|---:|---:|")
        for t, dp in idle[:30]:
            print(f"| {t:.2f} | {dp:+.3f} |")

    # For shoulder_lift specifically, show what cmd was doing at the time of each idle-drift event
    print("\n## shoulder_lift idle-drift events with surrounding cmd values:\n")
    j = 'shoulder_lift'
    events = per_joint_classified[j]
    idle = [(t, dp) for (t, dp, m) in events if not m]
    cmd_series = cmd_by_joint.get(j, [])
    print("| t_event (s) | dp_state | cmd_at_t-0.1 | cmd_at_t | cmd_at_t+0.1 | dcmd_per_100ms |")
    print("|---:|---:|---:|---:|---:|---:|")
    for t, dp in idle[:20]:
        def cmd_at(ts):
            # find nearest
            best = None
            best_dt = 1e9
            for tc, p in cmd_series:
                if abs(tc - ts) < best_dt:
                    best_dt = abs(tc - ts)
                    best = p
            return best
        c_before = cmd_at(t - 0.1)
        c_at = cmd_at(t)
        c_after = cmd_at(t + 0.1)
        if c_before is not None and c_after is not None:
            dcmd = c_after - c_before
            print(f"| {t:.2f} | {dp:+.3f} | {c_before:+.3f} | {c_at:+.3f} | {c_after:+.3f} | {dcmd:+.3f} |")


if __name__ == '__main__':
    main()
