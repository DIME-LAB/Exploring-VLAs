"""Deterministic collection parameters and timing acceptance rules (no ROS imports)."""
import math
import random

COLORS = ('red', 'green', 'blue')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper_joint')
# Training never uses these fixed seeds; evaluate each colour with each seed.
EVAL_SEEDS = tuple(range(900000, 900030))



def parameters(seed, color, split='train'):
    if split not in ('train', 'eval'):
        raise ValueError('Split must be train or eval')
    if split == 'eval' and seed not in EVAL_SEEDS:
        raise ValueError('Evaluation must use the fixed held-out seed set')
    if split == 'train' and seed in EVAL_SEEDS:
        raise ValueError('Reserved evaluation seed')
    if color not in COLORS:
        raise ValueError('Unknown task colour')
    rng = random.Random(seed)
    return dict(seed=seed, split=split, color=color, object=f'{color}_2x3_{rng.randrange(2)}',
                task=f'Pick a {color} lego and place it in {color} cup',
                approach_height_m=rng.uniform(.010, .030),
                approach_offset_m=[rng.uniform(-.004, .004), rng.uniform(-.004, .004)],
                speed_scale=rng.uniform(.85, 1.15), noise_std_rad=rng.uniform(.002, .006),
                noise_probability=.30, noise_seed=seed ^ 0xDA47,
                cup_seed=seed ^ 0xC0FFEE,
                cup_sampling=dict(radius_min=.20, radius_max=.27, angle_min_deg=-110.,
                                  angle_max_deg=-70., gap_min=.005, gap_max=.025))


def corrective_action(state, reference, max_step=.035, gain=.8):
    """Absolute expert setpoint, recomputed from the perturbed measured state."""
    return [q + max(-max_step, min(max_step, gain * (r - q)))
            for q, r in zip(state, reference)]


def timing_audit(rows, fps):
    errors = []
    if fps not in (20, 30):
        errors.append('fps must be 20 or 30')
    if len(rows) < fps * 2:
        return dict(passed=False, errors=['Fewer than two seconds of frames'], frames=len(rows))
    wall = [(b['wall_ns'] - a['wall_ns']) / 1e9 for a, b in zip(rows, rows[1:])]
    sim = [b['sim_ns'] - a['sim_ns'] for a, b in zip(rows, rows[1:])]
    if any(t <= 0 for t in wall):
        errors.append('Non-monotonic wall clock')
    if any(abs(t - round(1e9 / fps)) > 10 for t in sim):
        errors.append('Repeated or skipped sim camera frame')
    if any(abs(r['wrist_stamp_ns'] - r['top_stamp_ns']) > 1000 or r['sim_ns'] <= 0 for r in rows):
        errors.append('Camera stamps missing or not paired')
    if any(abs(r['joint_stamp_ns'] - r['sim_ns']) > 1000 for r in rows):
        errors.append('Joint state is not from the camera sim tick')
    if any(len(r[key]) != 6 for r in rows for key in ('state', 'action', 'executed_action', 'expert_reference')):
        errors.append('Joint feature shape differs from deployment')
    if any(max(abs(a - b) for a, b in zip(r['action'], corrective_action(r['state'], r['expert_reference']))) > 1e-9 for r in rows):
        errors.append('Recorded action differs from the clean corrective expert')
    if any(r['expert_observation_stamp_ns'] != r['sim_ns'] for r in rows):
        errors.append('Expert action is conditioned on a different observation tick')
    if any(r.get('state_source') != 'isaac_articulation' for r in rows):
        errors.append('Observation is not measured physical articulation feedback')
    if any(len(r.get('velocity', [])) != 6 or not all(math.isfinite(v) for v in r.get('velocity', [])) for r in rows):
        errors.append('Measured joint velocities are incomplete or non-finite')
    if any(abs(r.get('pose_stamp_ns', -1) - r['sim_ns']) > 1000 for r in rows):
        errors.append('Physical poses are not from the camera tick')
    application_latency = [r.get('application_stamp_ns', -1) - r['sim_ns'] for r in rows]
    if any(t < -10 or t > round(1e9 / fps) + 10 for t in application_latency):
        errors.append('Command application exceeds the declared one-tick contract')
    if any(abs(r.get('declared_application_stamp_ns', -1) - r['sim_ns']) > 10 for r in rows):
        errors.append('Command was not scheduled from its observation tick')
    if any(len(r.get('applied_command', [])) != 6 or any(abs(a-b) > 1e-6 for a,b in zip(r.get('applied_command', []),r['executed_action'])) for r in rows):
        errors.append('Applied physical drive target differs from executed command')
    planning_idle = sum(not r.get('executing', True) for r in rows) / len(rows)
    stationary = sum(max(abs(x - y) for x, y in zip(a['state'], b['state'])) < 1e-4
                     for a, b in zip(rows, rows[1:])) / (len(rows) - 1)
    if planning_idle > .03:
        errors.append('Planning/hold frames exceed 3%')
    if stationary > .05:
        errors.append('Stationary state frames exceed 5%')
    hz = (len(rows) - 1) / sum(wall) if sum(wall) > 0 else 0
    p95 = sorted(wall)[int(.95 * (len(wall) - 1))]
    if any(not all(math.isfinite(v) for v in r['action'] + r['state'] + r['executed_action']) for r in rows):
        errors.append('Non-finite joint data')
    return dict(passed=not errors, errors=errors, frames=len(rows), fps=fps,
                measured_wall_hz=hz, wall_gap_p95_ms=p95 * 1000,
                wall_gap_max_ms=max(wall) * 1000,
                sim_hz=1e9 * (len(rows) - 1) / sum(sim) if sum(sim) > 0 else 0,
                real_time_factor=sum(sim) / 1e9 / sum(wall),
                idle_frame_fraction=planning_idle, stationary_frame_fraction=stationary,
                command_latency_sim_ms_max=max((r['command_clock_ns'] - r['sim_ns']) / 1e6 for r in rows),
                application_latency_sim_ms_max=max(application_latency) / 1e6,
                state_image_stamp_skew_ns_max=max(abs(r['joint_stamp_ns'] - r['sim_ns']) for r in rows),
                noisy_steps=sum(r['action'] != r['executed_action'] for r in rows))
