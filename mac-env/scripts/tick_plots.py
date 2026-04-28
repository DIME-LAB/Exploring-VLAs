#!/usr/bin/env python3
"""Plot cmd vs state with LSB-events marked.

Markers:
  green dot  = LSB event during active commanded motion (expected)
  red star   = LSB event during idle-drift (= the "tick" the user sees)
"""
import json
import math
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


LSB_DEG = 360.0 / 4096.0


def load(path):
    cmds, states = [], []
    for line in open(path):
        r = json.loads(line)
        (cmds if r['kind'] == 'c' else states).append(r)
    cmds.sort(key=lambda r: r['t'])
    states.sort(key=lambda r: r['t'])
    if cmds:
        t0 = cmds[0]['t']
        for r in cmds: r['t'] -= t0
        for r in states: r['t'] -= t0
    return cmds, states


def per_joint(rows):
    out = defaultdict(lambda: ([], []))
    for r in rows:
        for n, p in zip(r['name'], r['pos']):
            ts, ps = out[n]
            ts.append(r['t'])
            ps.append(math.degrees(p))
    return out


def cmd_motion_at(cmd_series_ts_ps, t, half_window_s=0.5, threshold_deg=0.3):
    ts, ps = cmd_series_ts_ps
    nearby = [p for tt, p in zip(ts, ps) if abs(tt - t) <= half_window_s]
    if len(nearby) < 2:
        return False
    return (max(nearby) - min(nearby)) >= threshold_deg


def find_events(state_ts_ps, cmd_ts_ps):
    """Returns list of dicts: {t, p, dp, in_motion}."""
    ts, ps = state_ts_ps
    events = []
    for i in range(1, len(ts)):
        dp = ps[i] - ps[i - 1]
        if abs(dp) >= LSB_DEG:
            in_motion = cmd_motion_at(cmd_ts_ps, ts[i])
            events.append({
                't': ts[i], 'p': ps[i], 'dp': dp, 'in_motion': in_motion
            })
    return events


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/user_drive_v3.jsonl'
    cmds, states = load(path)
    cmd_j = per_joint(cmds)
    state_j = per_joint(states)
    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
              'wrist_flex', 'wrist_roll']

    # === Plot 1: full session per joint with events overlaid ===
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    for ax, j in zip(axes, joints):
        if j in cmd_j:
            ts, ps = cmd_j[j]
            ax.plot(ts, ps, 'b-', lw=0.6, label='cmd', alpha=0.7)
        if j in state_j:
            ts, ps = state_j[j]
            ax.plot(ts, ps, 'r-', lw=0.8, label='state')
            # Overlay LSB events
            events = find_events(state_j[j], cmd_j[j])
            during = [(e['t'], e['p']) for e in events if e['in_motion']]
            idle = [(e['t'], e['p']) for e in events if not e['in_motion']]
            if during:
                ax.scatter([t for t, _ in during], [p for _, p in during],
                           s=18, c='green', alpha=0.6, marker='o',
                           label=f'during motion ({len(during)})', zorder=3)
            if idle:
                ax.scatter([t for t, _ in idle], [p for _, p in idle],
                           s=120, c='red', marker='*', edgecolors='black',
                           linewidths=0.7,
                           label=f'IDLE-DRIFT TICK ({len(idle)})', zorder=4)
                # Vertical lines at each idle-drift tick
                for t, _ in idle:
                    ax.axvline(t, color='red', alpha=0.15, lw=1.0)
        ax.set_ylabel(f'{j}\n(deg)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(
        'Full session — cmd (blue) vs state (red); '
        'green ● = LSB event during motion; red ★ = idle-drift tick (visible to user)',
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig('/tmp/tick_plot_session_v2.png', dpi=120)
    print('saved /tmp/tick_plot_session_v2.png')

    # === Plot 2: zoom shoulder_lift idle-drift cluster ===
    if 'shoulder_lift' in cmd_j:
        fig2, ax2 = plt.subplots(figsize=(16, 7))
        ts, ps = cmd_j['shoulder_lift']
        ts_s, ps_s = state_j['shoulder_lift']
        events = find_events(state_j['shoulder_lift'], cmd_j['shoulder_lift'])
        t_lo, t_hi = 55.5, 58.0
        cmd_in = [(t, p) for t, p in zip(ts, ps) if t_lo <= t <= t_hi]
        state_in = [(t, p) for t, p in zip(ts_s, ps_s) if t_lo <= t <= t_hi]
        if cmd_in:
            ax2.plot([t for t, _ in cmd_in], [p for _, p in cmd_in],
                     'b-', lw=1.2, marker='.', markersize=4,
                     label='cmd (wire)', alpha=0.8)
        if state_in:
            ax2.plot([t for t, _ in state_in], [p for _, p in state_in],
                     'r-', lw=1.5, marker='o', markersize=5,
                     label='state (servo)', alpha=0.9)
        # Mark each LSB event in this window
        for e in events:
            if t_lo <= e['t'] <= t_hi:
                color = 'red' if not e['in_motion'] else 'green'
                ax2.axvline(e['t'], color=color, alpha=0.4, lw=1.5,
                            ls='--' if e['in_motion'] else '-')
                ax2.annotate(
                    f"Δ{e['dp']:+.2f}°",
                    xy=(e['t'], e['p']),
                    xytext=(e['t'] + 0.05, e['p'] + 0.3),
                    fontsize=8,
                    color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.5),
                )
        # LSB grid lines
        if state_in:
            avg_p = sum(p for _, p in state_in) / len(state_in)
            for k in range(-15, 15):
                ax2.axhline(avg_p + k * LSB_DEG, color='gray',
                            alpha=0.12, lw=0.4, ls=':')
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('shoulder_lift position (deg)')
        ax2.set_title(
            'shoulder_lift zoom — idle-drift tick cluster around t=56.6 '
            '(red lines = idle-drift LSB events)'
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig('/tmp/tick_plot_zoom_v2.png', dpi=120)
        print('saved /tmp/tick_plot_zoom_v2.png')

    # === Plot 3: error-band with events overlaid ===
    fig3, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    for ax, j in zip(axes, joints):
        if j not in cmd_j or j not in state_j:
            continue
        cts, cps = cmd_j[j]
        sts, sps = state_j[j]
        # interp state at each cmd timestamp
        errs = []
        si = 0
        for ti, pi in zip(cts, cps):
            while si < len(sts) - 1 and sts[si + 1] < ti:
                si += 1
            if si < len(sts) - 1 and sts[si + 1] > sts[si]:
                t0, p0 = sts[si], sps[si]
                t1, p1 = sts[si + 1], sps[si + 1]
                s = max(0.0, min(1.0, (ti - t0) / (t1 - t0)))
                errs.append(pi - (p0 + s * (p1 - p0)))
            else:
                errs.append(0.0)
        ax.plot(cts, errs, 'g-', lw=0.6)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(+0.88, color='r', lw=0.7, ls='--', alpha=0.5,
                   label='servo deadband ±0.88°')
        ax.axhline(-0.88, color='r', lw=0.7, ls='--', alpha=0.5)
        # Mark idle-drift events
        events = find_events(state_j[j], cmd_j[j])
        for e in events:
            if not e['in_motion']:
                ax.axvline(e['t'], color='red', alpha=0.4, lw=1.5)
        ax.set_ylabel(f'{j}\ncmd−state (deg)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
    axes[-1].set_xlabel('time (s)')
    fig3.suptitle(
        'Wire cmd minus servo state. Vertical red lines = idle-drift ticks.',
        fontsize=11
    )
    fig3.tight_layout()
    fig3.savefig('/tmp/tick_plot_error_band_v2.png', dpi=120)
    print('saved /tmp/tick_plot_error_band_v2.png')


if __name__ == '__main__':
    main()
