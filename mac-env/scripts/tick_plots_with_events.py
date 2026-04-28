#!/usr/bin/env python3
"""Plot cmd vs state with TRAJECTORY events from shim log overlaid as
labeled vertical lines. Uses a heuristic to align shim's unix-epoch
timestamps with the recorder's monotonic clock by matching the first
significant cmd motion."""
import json
import math
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


LSB_DEG = 360.0 / 4096.0
LOG_LINE_RE = re.compile(
    r'^\[(?:INFO|WARN)\] \[(\d+)\.(\d+)\] \[trajectory_shim\]: (.*)$'
)


def load_recorder(path):
    cmds, states = [], []
    for line in open(path):
        r = json.loads(line)
        (cmds if r['kind'] == 'c' else states).append(r)
    cmds.sort(key=lambda r: r['t'])
    states.sort(key=lambda r: r['t'])
    return cmds, states


def load_shim_events(path, t_min_unix, t_max_unix):
    """Parse shim log for trajectory start/end events within a time window.
    Returns list of (t_unix, label_string)."""
    events = []
    for line in open(path):
        m = LOG_LINE_RE.match(line.strip())
        if not m:
            continue
        t_unix = int(m.group(1)) + int(m.group(2)) / 1e9
        if not (t_min_unix <= t_unix <= t_max_unix):
            continue
        body = m.group(3)
        # arm starts
        m2 = re.match(r'\[(arm|gripper)\] (\d+) waypoints, scaled duration ([\d.]+)s', body)
        if m2:
            label = f"{m2.group(1)} START ({m2.group(2)}wp, {m2.group(3)}s)"
            events.append((t_unix, label, 'start'))
            continue
        m2 = re.match(r'\[(arm|gripper)\] reached goal \(max_err=([\d.]+)', body)
        if m2:
            label = f"{m2.group(1)} END (err={m2.group(2)}°)"
            events.append((t_unix, label, 'end_ok'))
            continue
        m2 = re.match(r'\[(arm|gripper)\] tolerance violated: max_err=([\d.]+)', body)
        if m2:
            label = f"{m2.group(1)} ABORT (err={m2.group(2)}°)"
            events.append((t_unix, label, 'end_fail'))
            continue
        m2 = re.match(r'\[(arm|gripper)\] (preempted|cancelled)', body)
        if m2:
            events.append((t_unix, f"{m2.group(1)} {m2.group(2)}", 'preempt'))
    return events


def per_joint(rows):
    out = defaultdict(lambda: ([], []))
    for r in rows:
        for n, p in zip(r['name'], r['pos']):
            ts, ps = out[n]
            ts.append(r['t'])
            ps.append(math.degrees(p))
    return out


def find_first_big_motion(cmd_by_joint, threshold_deg=2.0):
    """Find earliest cmd timestamp where any joint's cmd shifted by
    >threshold over 0.5s, indicating a real trajectory start."""
    best_t = None
    for ts, ps in cmd_by_joint.values():
        for i in range(1, len(ts)):
            # look back 0.5s
            j = i
            while j > 0 and ts[i] - ts[j] < 0.5:
                j -= 1
            window = ps[j:i + 1]
            if len(window) < 2:
                continue
            if (max(window) - min(window)) >= threshold_deg:
                if best_t is None or ts[j] < best_t:
                    best_t = ts[j]
                break
    return best_t


def main():
    rec_path = '/tmp/user_drive_v3.jsonl'
    log_path = '/tmp/traj_shim.log'

    cmds, states = load_recorder(rec_path)
    if not cmds:
        print('no cmds in recorder file')
        return

    rec_t0 = cmds[0]['t']  # monotonic seconds at first cmd
    rec_t_end = cmds[-1]['t']

    cmd_j_raw = per_joint(cmds)
    state_j_raw = per_joint(states)

    # Find the first big motion in the recorder
    first_motion_t_mono = find_first_big_motion(cmd_j_raw)
    if first_motion_t_mono is None:
        print('no big motion found')
        return

    # Find the matching first "X waypoints" line in shim log near that moment.
    # We'll search the shim log from just before recording start to just after.
    # Recorder doesn't have a unix-epoch reference, so we estimate: the file's
    # mtime (when last write happened) ≈ rec_t_end in unix-epoch. Use stat.
    import os
    rec_mtime_unix = os.path.getmtime(rec_path)
    # rec_mtime_unix ≈ time when last cmd was written. So unix_offset ≈
    # rec_mtime_unix - rec_t_end (mapping monotonic to unix).
    candidate_offset = rec_mtime_unix - rec_t_end
    # Now first_motion in unix-epoch:
    first_motion_unix_estimate = first_motion_t_mono + candidate_offset

    # Find shim log "X waypoints" event closest to that estimate.
    # Look in a +/- 30s window.
    log_events_near = load_shim_events(
        log_path,
        first_motion_unix_estimate - 30,
        first_motion_unix_estimate + 30,
    )
    starts = [e for e in log_events_near if e[2] == 'start']
    if not starts:
        print('no start events found near estimated time; falling back to mtime offset')
        offset = candidate_offset
    else:
        # Closest start to our motion estimate
        closest = min(starts, key=lambda e: abs(e[0] - first_motion_unix_estimate))
        # Calibrate: closest[0] (unix) should equal first_motion_t_mono + offset
        # → offset = closest[0] - first_motion_t_mono
        offset = closest[0] - first_motion_t_mono
        print(f'aligned: shim event "{closest[1]}" @ unix {closest[0]:.2f} → '
              f'recorder t={first_motion_t_mono - rec_t0:.2f}s')
        print(f'offset (unix - monotonic): {offset:.2f}s')

    # Now load all shim events in the recording window (in unix epoch, then
    # convert back to recorder-relative seconds by subtracting offset+rec_t0)
    rec_window_unix_start = rec_t0 + offset
    rec_window_unix_end = rec_t_end + offset
    all_events_unix = load_shim_events(
        log_path, rec_window_unix_start, rec_window_unix_end
    )
    # Convert to recorder-relative time (t=0 at first cmd)
    events_rel = [(t_unix - offset - rec_t0, lbl, kind)
                  for (t_unix, lbl, kind) in all_events_unix]
    print(f'\n{len(events_rel)} trajectory events in recording window:')
    for t, lbl, kind in events_rel:
        print(f'  t={t:6.2f}s  {kind:10s}  {lbl}')

    # === Plot: full session per joint with shim events as vertical lines ===
    # Zero recorder times for plotting
    for r in cmds: r['t'] -= rec_t0
    for r in states: r['t'] -= rec_t0
    cmd_j = per_joint(cmds)
    state_j = per_joint(states)

    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll']

    fig, axes = plt.subplots(5, 1, figsize=(18, 13), sharex=True)
    color_for = {'start': 'green', 'end_ok': 'blue', 'end_fail': 'orange',
                 'preempt': 'gray'}
    for ax, j in zip(axes, joints):
        if j in cmd_j:
            ts, ps = cmd_j[j]
            ax.plot(ts, ps, 'b-', lw=0.6, label='cmd', alpha=0.7)
        if j in state_j:
            ts, ps = state_j[j]
            ax.plot(ts, ps, 'r-', lw=0.8, label='state')
        # Overlay shim events as vertical lines with labels (only on top axis)
        for t, lbl, kind in events_rel:
            ax.axvline(t, color=color_for.get(kind, 'k'),
                       alpha=0.5, lw=1.0, ls='--' if kind == 'preempt' else '-')
        ax.set_ylabel(f'{j}\n(deg)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    # On the top axis, label each event
    top_ax = axes[0]
    y_top = top_ax.get_ylim()[1]
    for i, (t, lbl, kind) in enumerate(events_rel):
        if kind in ('start', 'end_ok', 'end_fail'):
            top_ax.annotate(
                lbl,
                xy=(t, y_top),
                xytext=(t, y_top + (i % 3) * 1.5),
                fontsize=7,
                color=color_for.get(kind, 'k'),
                rotation=70,
                ha='center',
                va='bottom',
            )
    axes[-1].set_xlabel('time (s, since first cmd)')
    fig.suptitle(
        'Full session — vertical lines = shim trajectory events  '
        '(green=START, blue=END ok, orange=ABORT, dashed gray=preempt)',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig('/tmp/tick_plot_with_events.png', dpi=120)
    print('\nsaved /tmp/tick_plot_with_events.png')


if __name__ == '__main__':
    main()
