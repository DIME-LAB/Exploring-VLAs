#!/usr/bin/env python3
"""System identification: fit underdamped second-order model per joint
from a JSONL trace produced by motion_recorder.py.

Model (post-step):
  y(t) = y_∞ + A·exp(-ζ·ω_n·t)·cos(ω_d·t + φ)
  where ω_d = ω_n·√(1-ζ²)

Identification window: the FINAL hold phase (hold_B). The setpoint is
constant during this window; any deviation is the system's own response
to the (just-completed) step input.

Outputs:
  - per-joint {omega_n, zeta, T_oscillation, y_inf, residual_rms_deg}
  - matplotlib figure showing fit overlay per joint
"""
import argparse
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper_joint']


def underdamped_2nd_order(t, A, omega_n, zeta, phi, y_inf):
    """Underdamped second-order step response (assumes 0 < ζ < 1)."""
    if zeta >= 1.0 or zeta < 0:
        return np.full_like(t, y_inf)
    omega_d = omega_n * math.sqrt(max(1e-9, 1.0 - zeta * zeta))
    return y_inf + A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)


def fit_joint(times, actual, y_target_setpoint):
    """Fit underdamped 2nd-order to (times, actual). Returns dict."""
    t = np.array(times) - times[0]
    y = np.array(actual)
    if len(t) < 30:
        return None

    # Initial parameter guesses
    y0 = float(y[0])
    y_inf_guess = float(np.mean(y[-len(y) // 4:]))
    A_guess = y0 - y_inf_guess
    # Estimate oscillation period from zero-crossings of (y - y_inf)
    centered = y - y_inf_guess
    zc = []
    for i in range(1, len(centered)):
        if centered[i - 1] * centered[i] < 0:
            zc.append(t[i])
    if len(zc) >= 2:
        T_guess = 2 * (zc[1] - zc[0])
    else:
        T_guess = (t[-1] - t[0]) / 4.0
    omega_n_guess = 2 * math.pi / max(T_guess, 0.01)
    # Estimate damping from amplitude decay (envelope)
    if abs(A_guess) > 1e-6 and len(y) > 50:
        # Sample envelope at ~quarter and three-quarter
        env_early = np.abs(y[:len(y) // 4] - y_inf_guess).max()
        env_late = np.abs(y[3 * len(y) // 4:] - y_inf_guess).max()
        if env_early > 1e-6 and env_late > 1e-6 and env_late < env_early:
            decay_ratio = env_late / env_early
            zeta_guess = max(0.01, min(0.5,
                            -math.log(decay_ratio) / (omega_n_guess * (t[-1] - t[0]) * 0.5)))
        else:
            zeta_guess = 0.1
    else:
        zeta_guess = 0.1

    p0 = [A_guess, omega_n_guess, zeta_guess, 0.0, y_inf_guess]
    bounds = ([-math.pi, 0.5, 0.001, -2 * math.pi, -math.pi],
              [math.pi,  500.0, 0.99,  2 * math.pi,  math.pi])
    try:
        popt, _ = curve_fit(underdamped_2nd_order, t, y, p0=p0, bounds=bounds,
                            maxfev=5000)
    except Exception as e:
        return {'error': str(e), 'p0': p0}
    A, omega_n, zeta, phi, y_inf = popt
    y_fit = underdamped_2nd_order(t, *popt)
    residual = y - y_fit
    rms = float(np.sqrt(np.mean(residual ** 2)))
    omega_d = omega_n * math.sqrt(max(1e-9, 1.0 - zeta * zeta))
    T_osc = 2 * math.pi / omega_d if omega_d > 0 else float('inf')

    return {
        'A_rad': float(A),
        'omega_n': float(omega_n),
        'zeta': float(zeta),
        'phi': float(phi),
        'y_inf_rad': float(y_inf),
        'y_inf_deg': math.degrees(float(y_inf)),
        'T_osc_s': T_osc,
        'residual_rms_deg': math.degrees(rms),
        'amplitude_initial_deg': math.degrees(abs(A)),
        'sag_deg': math.degrees(y_target_setpoint - y_inf) if y_target_setpoint is not None else None,
    }


def load_run(jsonl_path):
    samples = [json.loads(line) for line in open(jsonl_path)]
    return samples


def find_hold_b_window(samples, hold_duration_s=4.0, move_duration_s=3.0):
    """Heuristic: hold_B is the FINAL hold_duration seconds of the run."""
    t_end = samples[-1]['t']
    t_start = max(0.0, t_end - hold_duration_s)
    return [s for s in samples if s['t'] >= t_start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSONL from motion_recorder')
    ap.add_argument('--hold-duration', type=float, default=4.0)
    ap.add_argument('--move-duration', type=float, default=3.0)
    ap.add_argument('--output-prefix', required=True)
    args = ap.parse_args()

    samples = load_run(args.input)
    if not samples:
        print("FAIL: empty trace")
        return 1

    # Use the post-move-B window as the step-response data.
    # The cmd is constant during hold (= pose_b), so any actual variation
    # is the response to the just-finished move.
    # Take a wider window: from the start of move_A_to_B to the end,
    # but feed only the *post-step-end* portion to the fitter.
    # (We want where cmd has stabilized but actual hasn't yet.)
    t_end = samples[-1]['t']
    # The hold_B window:
    hold_b_start = t_end - args.hold_duration
    hold_b_samples = [s for s in samples if s['t'] >= hold_b_start]
    if len(hold_b_samples) < 30:
        print(f"FAIL: hold_B window has only {len(hold_b_samples)} samples")
        return 1

    times = [s['t'] for s in hold_b_samples]
    # Per-joint setpoint at hold_B (cmd is constant in this window)
    sp_per_joint = {}
    actual_per_joint = {n: [] for n in JOINT_NAMES}
    for s in hold_b_samples:
        for n in JOINT_NAMES:
            actual_per_joint[n].append(s['actual'].get(n, np.nan))
            sp_per_joint[n] = s['cmd'].get(n, np.nan)  # last value wins

    results = {}
    fig, axs = plt.subplots(len(JOINT_NAMES), 1, figsize=(12, 2.0 * len(JOINT_NAMES)),
                            sharex=True)
    fig.suptitle(f'System ID — hold_B window (last {args.hold_duration:.1f}s)',
                 fontsize=11)
    for i, j in enumerate(JOINT_NAMES):
        actual_rad = actual_per_joint[j]
        sp_rad = sp_per_joint[j]
        if any(math.isnan(v) for v in actual_rad):
            results[j] = {'note': 'NaN in actual'}
            continue
        result = fit_joint(times, actual_rad, sp_rad)
        results[j] = result

        # Plot
        t_arr = np.array(times) - times[0]
        actual_arr = np.array(actual_rad)
        axs[i].plot(t_arr, np.degrees(actual_arr), 'o', color='C3', markersize=2,
                    label='actual', alpha=0.6)
        axs[i].axhline(math.degrees(sp_rad), color='C0', linestyle='--',
                       linewidth=1.0, label=f'cmd ({math.degrees(sp_rad):+.1f}°)')
        if result and 'omega_n' in result:
            popt = (result['A_rad'], result['omega_n'], result['zeta'],
                    result['phi'], result['y_inf_rad'])
            t_dense = np.linspace(t_arr[0], t_arr[-1], 500)
            y_fit = underdamped_2nd_order(t_dense, *popt)
            axs[i].plot(t_dense, np.degrees(y_fit), '-', color='black',
                        linewidth=1.0,
                        label=(f'fit ω_n={result["omega_n"]:.1f} rad/s '
                               f'ζ={result["zeta"]:.3f} '
                               f'T={result["T_osc_s"]*1000:.0f}ms'))
        axs[i].set_ylabel(f'{j}\n(deg)', fontsize=8)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best', fontsize=7)
    axs[-1].set_xlabel('t (s, relative)')
    plt.tight_layout()
    fig_path = args.output_prefix + '.sysid.png'
    plt.savefig(fig_path, dpi=110)
    plt.close(fig)
    print(f"Plot: {fig_path}")

    # Save numerical results
    json_path = args.output_prefix + '.sysid.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {json_path}")

    # Print summary
    print("\n--- System ID summary ---")
    print(f"{'Joint':<14} {'ω_n (rad/s)':<12} {'ζ':<8} {'T_osc (ms)':<12} {'sag (°)':<10} {'fit RMS (°)':<10}")
    for j, r in results.items():
        if not r or 'omega_n' not in r:
            note = (r or {}).get('note') or (r or {}).get('error', 'no fit')
            print(f"{j:<14} {note}")
            continue
        T_ms = r['T_osc_s'] * 1000.0
        print(f"{j:<14} {r['omega_n']:<12.2f} {r['zeta']:<8.3f} {T_ms:<12.0f} "
              f"{r['sag_deg']:<10.2f} {r['residual_rms_deg']:<10.3f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
