#!/usr/bin/env python3
"""Motion recorder + driver + plotter — single-process, time-aligned.

Drives the SO-ARM101 follower through a 2-pose test sequence while
simultaneously recording commanded vs achieved joint positions. Saves
data to JSONL and produces a per-joint matplotlib plot showing:
  - top subplot: cmd (dashed) vs actual (solid) over time, all joints
  - one row per joint: cmd vs actual + error trace

Trajectory generation: **quintic** (zero velocity AND acceleration at
endpoints). Smoother than linear; doesn't excite the underdamped modes
the way step-discontinuous setpoints do.

Pose presets:
  zero  — all joints = 0  (URDF rest, arm fully extended horizontal)
  home  — grasp_home: wrist_flex=π/2, wrist_roll=-π/2, others=0

Phases of one run:
  Phase 0: t∈[0, t1)        : settle (publish current pose, capture)
  Phase 1: t∈[t1, t1+T)     : move CURRENT → pose-A
  Phase 2: t∈[t1+T, t1+T+H) : hold at pose-A
  Phase 3: t∈[…, …+T)       : move pose-A → pose-B
  Phase 4: t∈[…, …+H)       : hold at pose-B

Usage:
  python3 motion_recorder.py \\
      --pose-a zero --pose-b home \\
      --move-duration 3.0 --hold-duration 4.0 \\
      --output /tmp/run_001
"""
import argparse
import json
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')   # no GUI; we only save PNGs
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper_joint']

POSES = {
    'zero': {n: 0.0 for n in JOINT_NAMES},
    'home': {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': math.pi / 2,
        'wrist_roll': -math.pi / 2,
        'gripper_joint': 0.0,
    },
}


def quintic(a):
    """Quintic minimum-jerk profile s(a) for a∈[0,1].

    s(0) = 0, s(1) = 1, s'(0) = s'(1) = 0, s''(0) = s''(1) = 0.
    Formula: 10a³ - 15a⁴ + 6a⁵.
    """
    return 10.0 * a**3 - 15.0 * a**4 + 6.0 * a**5


def interp_pose(start, end, alpha):
    s = quintic(max(0.0, min(1.0, alpha)))
    return {j: start[j] + s * (end[j] - start[j]) for j in start}


class Recorder(Node):
    """Subscribes to /joint_states + /joint_commands, publishes target.

    publish_topic='/joint_commands' → direct (no FF layer in the loop)
    publish_topic='/joint_targets'  → gravity_comp picks it up, applies
                                       FF + integrator, publishes /joint_commands

    Records every (time, last_cmd, last_actual) snapshot to self.samples.
    """

    def __init__(self, publish_topic='/joint_commands'):
        self._publish_topic = publish_topic
        super().__init__('motion_recorder')
        self.last_actual = {}
        self.last_cmd = {}
        self.samples = []
        # Publish target. /joint_commands goes directly to jsr (no FF/integrator).
        # /joint_targets goes through gravity_comp (which corrects then publishes
        # to /joint_commands itself).
        publish_topic = getattr(self, '_publish_topic', '/joint_commands')
        self.cmd_pub = self.create_publisher(JointState, publish_topic, 50)
        self.create_subscription(JointState, '/joint_states', self._on_state, 50)
        # Always observe /joint_commands (the actual setpoint sent to jsr) — even
        # when we publish to /joint_targets and gravity_comp transforms it, the
        # observer still records what the bus actually saw.
        self.create_subscription(JointState, '/joint_commands', self._on_cmd, 50)

    def _on_state(self, msg):
        for i, n in enumerate(msg.name):
            self.last_actual[n] = msg.position[i]

    def _on_cmd(self, msg):
        for i, n in enumerate(msg.name):
            self.last_cmd[n] = msg.position[i]

    def publish_setpoint(self, target_dict):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(target_dict.keys())
        msg.position = list(target_dict.values())
        self.cmd_pub.publish(msg)

    def snapshot(self, t):
        """Record a single time-aligned (cmd, actual) sample."""
        self.samples.append({
            't': t,
            'cmd': dict(self.last_cmd),
            'actual': dict(self.last_actual),
        })


def run(args):
    rclpy.init()
    rec = Recorder(publish_topic=args.publish_topic)

    # Wait for first /joint_states
    deadline = time.time() + 5.0
    while time.time() < deadline and not rec.last_actual:
        rclpy.spin_once(rec, timeout_sec=0.1)
    if not rec.last_actual:
        print("FAIL: no /joint_states received in 5s — is jsr_write running?")
        rclpy.shutdown()
        return 1

    # Snapshot starting pose, filtered to known joints (drop any extras).
    start_pose = {n: rec.last_actual.get(n, 0.0) for n in JOINT_NAMES}
    pose_a = POSES[args.pose_a]
    pose_b = POSES[args.pose_b]
    move_T = args.move_duration
    hold_T = args.hold_duration
    rate = args.rate_hz
    dt = 1.0 / rate

    print(f"Starting pose: { {k: round(math.degrees(v),1) for k,v in start_pose.items()} }")
    print(f"Pose A ({args.pose_a}): { {k: round(math.degrees(v),1) for k,v in pose_a.items()} }")
    print(f"Pose B ({args.pose_b}): { {k: round(math.degrees(v),1) for k,v in pose_b.items()} }")

    t0 = time.time()
    phase_log = []

    def run_phase(name, segment_target_fn, duration):
        """Loop for `duration` s, publishing segment_target_fn(alpha)
        at `rate` Hz, recording snapshots."""
        steps = int(duration * rate)
        phase_log.append((name, time.time() - t0))
        for i in range(steps + 1):
            alpha = i / steps if steps > 0 else 1.0
            tgt = segment_target_fn(alpha)
            rec.publish_setpoint(tgt)
            rclpy.spin_once(rec, timeout_sec=0.0)
            rec.snapshot(time.time() - t0)
            time.sleep(dt)

    # Phase 0: settle (1.0 s) — publish current pose, capture data
    run_phase('settle_pre',  lambda a: start_pose, 1.0)

    # Phase 1: move start → A
    run_phase('move_start_to_A',
              lambda a: interp_pose(start_pose, pose_a, a),
              move_T)
    # Phase 2: hold at A
    run_phase('hold_A', lambda a: pose_a, hold_T)

    # Phase 3: move A → B
    run_phase('move_A_to_B',
              lambda a: interp_pose(pose_a, pose_b, a),
              move_T)
    # Phase 4: hold at B
    run_phase('hold_B', lambda a: pose_b, hold_T)

    rec.destroy_node()
    rclpy.shutdown()

    # Save JSONL
    jsonl_path = args.output + '.jsonl'
    with open(jsonl_path, 'w') as f:
        for s in rec.samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(rec.samples)} samples to {jsonl_path}")

    # Plot
    png_path = args.output + '.png'
    plot_run(rec.samples, phase_log, png_path,
             title=f'{args.pose_a} → {args.pose_b}  (P=16 D=32 I=0)')
    print(f"Plot: {png_path}")

    # Print quick summary
    summary = compute_summary(rec.samples, phase_log)
    print("\n--- Summary ---")
    for j, s in summary.items():
        print(f"  {j:<14} steady_err_A={s['steady_err_A']:+.2f}°  "
              f"steady_err_B={s['steady_err_B']:+.2f}°  "
              f"vib_std_B={s['vib_std_B']:.3f}°  "
              f"max_track_err={s['max_track_err']:.2f}°")

    summary_path = args.output + '.summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return 0


def compute_summary(samples, phase_log):
    """Per-joint stats across the run.
       steady_err_X = mean(actual - cmd) over last 50% of hold_X
       vib_std_X    = stdev(actual) over last 50% of hold_X
       max_track_err = max |actual - cmd| during any move phase
    """
    # Find phase boundaries
    phases = dict(phase_log)
    t_settle = phases['settle_pre']
    t_moveA = phases['move_start_to_A']
    t_holdA = phases['hold_A']
    t_moveB = phases['move_A_to_B']
    t_holdB = phases['hold_B']
    t_end = samples[-1]['t']

    def slice_t(t_lo, t_hi):
        return [s for s in samples if t_lo <= s['t'] < t_hi]

    holdA_samples = slice_t(t_holdA, t_moveB)
    holdB_samples = slice_t(t_holdB, t_end)
    moveA_samples = slice_t(t_moveA, t_holdA)
    moveB_samples = slice_t(t_moveB, t_holdB)

    def steady_window(s_list):
        n = len(s_list)
        return s_list[n // 2:] if n > 4 else s_list

    out = {}
    for j in JOINT_NAMES:
        # Steady error at A
        sa = steady_window(holdA_samples)
        eA = [math.degrees(s['actual'].get(j, 0) - s['cmd'].get(j, 0))
              for s in sa if j in s['actual']]
        steady_err_A = float(np.mean(eA)) if eA else 0.0

        # Steady error at B + vibration std
        sb = steady_window(holdB_samples)
        actB = [math.degrees(s['actual'][j]) for s in sb if j in s['actual']]
        cmdB = [math.degrees(s['cmd'].get(j, 0)) for s in sb if j in s['actual']]
        steady_err_B = float(np.mean(np.array(actB) - np.array(cmdB))) if actB else 0.0
        vib_std_B = float(np.std(actB)) if actB else 0.0

        # Max tracking error during moves
        track = []
        for s in moveA_samples + moveB_samples:
            if j in s['actual'] and j in s['cmd']:
                track.append(abs(math.degrees(s['actual'][j] - s['cmd'][j])))
        max_track_err = float(max(track)) if track else 0.0

        out[j] = {
            'steady_err_A': steady_err_A,
            'steady_err_B': steady_err_B,
            'vib_std_B': vib_std_B,
            'max_track_err': max_track_err,
        }
    return out


def plot_run(samples, phase_log, path, title):
    """Per-joint cmd vs actual + error trace."""
    if not samples:
        return
    t = np.array([s['t'] for s in samples])
    fig, axs = plt.subplots(len(JOINT_NAMES), 2,
                            figsize=(14, 2.0 * len(JOINT_NAMES)),
                            sharex=True)
    fig.suptitle(title, fontsize=11)
    for i, j in enumerate(JOINT_NAMES):
        cmd = np.array([math.degrees(s['cmd'].get(j, np.nan)) for s in samples])
        act = np.array([math.degrees(s['actual'].get(j, np.nan)) for s in samples])
        axs[i, 0].plot(t, cmd, '--', color='C0', linewidth=1.0, label='cmd')
        axs[i, 0].plot(t, act, '-',  color='C3', linewidth=1.0, label='actual')
        axs[i, 0].set_ylabel(f'{j}\n(deg)', fontsize=8)
        axs[i, 0].grid(True, alpha=0.3)
        axs[i, 0].legend(loc='upper right', fontsize=7)

        # Error trace
        err = act - cmd
        axs[i, 1].plot(t, err, '-', color='C2', linewidth=0.8)
        axs[i, 1].axhline(0, color='k', linewidth=0.4)
        axs[i, 1].set_ylabel(f'err (°)', fontsize=8)
        axs[i, 1].grid(True, alpha=0.3)

        # Phase boundaries
        for label, t_phase in phase_log:
            for ax in axs[i]:
                ax.axvline(t_phase, color='gray', linewidth=0.4, alpha=0.5)

    axs[-1, 0].set_xlabel('t (s)')
    axs[-1, 1].set_xlabel('t (s)')

    # Annotate phase boundaries on top
    for ax in axs[0]:
        ax.set_xlim(t[0], t[-1])
    for label, t_phase in phase_log:
        axs[0, 0].text(t_phase, axs[0, 0].get_ylim()[1], label,
                       rotation=90, fontsize=6, va='top', alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose-a', choices=list(POSES.keys()), default='zero')
    ap.add_argument('--pose-b', choices=list(POSES.keys()), default='home')
    ap.add_argument('--move-duration', type=float, default=3.0)
    ap.add_argument('--hold-duration', type=float, default=4.0)
    ap.add_argument('--rate-hz', type=float, default=50.0)
    ap.add_argument('--publish-topic', default='/joint_commands',
                    help='/joint_commands = direct to jsr; '
                         '/joint_targets = through gravity_comp')
    ap.add_argument('--output', required=True,
                    help='Path prefix; outputs <prefix>.jsonl, .png, .summary.json')
    args = ap.parse_args()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
