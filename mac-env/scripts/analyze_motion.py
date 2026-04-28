#!/usr/bin/env python3
"""Per-joint motion-quality analysis from a record_motion_topics.py JSONL.

For each detected motion segment (burst of /joint_commands separated by
>500 ms gaps), reports per joint:
  - duration, peak |error|, RMS error during motion
  - settling time after motion-end (until |err| < 0.5° for 200 ms)
  - residual hold error (mean |err| in last 0.3s of observation window)
  - dominant oscillation frequency in the error signal (3-25 Hz bins)

Use case: empirical proof that a smoothness fix works. Capture before
+ after JSONLs and compare hold_e / settle / f_osc columns.

Usage:
  pixi run --manifest-path /tmp/mac-env/pixi.toml python3 \\
      mac-env/scripts/analyze_motion.py /tmp/motion_record.jsonl

Pairs with: mac-env/scripts/record_motion_topics.py
"""
import json, math, sys
from collections import defaultdict

REC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/motion_record_v2.jsonl'
ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
SETTLE_THR = math.radians(0.5)

buckets = {'commands': defaultdict(list), 'states': defaultdict(list)}
for line in open(REC):
    r = json.loads(line)
    if r['kind'] not in buckets: continue
    for n, p in zip(r['name'], r['pos']):
        if n in ARM:
            buckets[r['kind']][n].append((r['t'], p))

t0 = min(buckets['states']['shoulder_pan'][0][0],
         buckets['commands']['shoulder_pan'][0][0])
for k in buckets:
    for n in buckets[k]:
        buckets[k][n] = [(t - t0, p) for t, p in buckets[k][n]]

# Segments: bursts of /joint_commands (gap > 500 ms = new segment)
cmd_t = sorted({t for n in ARM for t, _ in buckets['commands'][n]})
segs = []
seg_s = None; prev = None
for t in cmd_t:
    if prev is None or (t - prev) > 0.5:
        if seg_s is not None: segs.append((seg_s, prev))
        seg_s = t
    prev = t
if seg_s is not None: segs.append((seg_s, prev))

print(f'{len(segs)} motion segments:')
for i, (s, e) in enumerate(segs):
    print(f'  seg {i}: t=[{s:.2f}, {e:.2f}]  active={e-s:.2f}s')
print()

def lerp(samples, t):
    if not samples: return None
    if t <= samples[0][0]: return samples[0][1]
    if t >= samples[-1][0]: return samples[-1][1]
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid][0] <= t: lo = mid
        else: hi = mid
    t0_, p0_ = samples[lo]; t1_, p1_ = samples[hi]
    return p0_ + (t - t0_) / (t1_ - t0_) * (p1_ - p0_)

print(f'{"seg":<3} {"joint":<14} {"act_dur":>7} {"peak_e":>7} {"rms_e":>6} '
      f'{"settle":>7} {"hold_e":>7} {"f_osc":>6}')
print('-' * 65)
for sidx, (s, e) in enumerate(segs):
    dur = e - s
    settle_window = 1.5
    for n in ARM:
        cmds = sorted(buckets['commands'][n])
        states = sorted(buckets['states'][n])
        t_grid = [s + 0.01 * i for i in range(int((dur + settle_window) * 100))]
        cmd_v = [lerp(cmds, t) for t in t_grid]
        act_v = [lerp(states, t) for t in t_grid]
        if any(c is None or a is None for c, a in zip(cmd_v, act_v)):
            continue
        err = [c - a for c, a in zip(cmd_v, act_v)]
        in_motion = [t < s + dur for t in t_grid]
        m_err = [e_ for e_, m in zip(err, in_motion) if m]
        if not m_err: continue
        peak_e = max(abs(x) for x in m_err)
        rms_e = math.sqrt(sum(x*x for x in m_err) / len(m_err))
        post_idx = [i for i, t in enumerate(t_grid) if t >= s + dur]
        settle_time = float('nan')
        for k_idx in post_idx:
            if all(abs(err[j]) < SETTLE_THR
                   for j in range(k_idx, min(k_idx + 20, len(err)))):
                settle_time = t_grid[k_idx] - (s + dur)
                break
        hold_e = sum(abs(err[i]) for i in post_idx[-30:]) / max(len(post_idx[-30:]), 1)
        n_pts = len(m_err)
        f_osc = 0
        if n_pts > 16:
            best_mag = 0.0; best_f = 0
            for hz in [3, 5, 7, 10, 15, 20, 25]:
                w = 2 * math.pi * hz
                re = sum(m_err[i] * math.cos(w * t_grid[i]) for i in range(n_pts))
                im = sum(m_err[i] * math.sin(w * t_grid[i]) for i in range(n_pts))
                mag = math.sqrt(re*re + im*im) / n_pts
                if mag > best_mag:
                    best_mag = mag; best_f = hz
            if best_mag > math.radians(0.5):
                f_osc = best_f
        print(f'{sidx:<3} {n:<14} {dur:>6.2f}s '
              f'{math.degrees(peak_e):>6.2f}° '
              f'{math.degrees(rms_e):>5.2f}° '
              f'{settle_time:>6.2f}s '
              f'{math.degrees(hold_e):>6.2f}° '
              f'{f_osc:>4d}Hz')
    print()
