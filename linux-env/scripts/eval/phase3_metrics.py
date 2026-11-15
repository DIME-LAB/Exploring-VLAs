"""Colour-aware physical outcome scoring for Phase3; timestamps are sim seconds."""
from __future__ import annotations

import math
import sys
from pathlib import Path

COLORS = ('red', 'green', 'blue')


class PhysicalScore:
    def __init__(self, color, initial, fk):
        # Reuse the collector's whole-block, cup-local physical geometry test.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'collect'))
        from success import contained
        self.contained = contained
        self.color, self.fk = color, fk
        self.starts = {k: v['position'][2] for k, v in initial['bodies'].items() if not k.startswith('cup_')}
        self.begin = None
        self.last = None
        self.lift_since, self.inside_since = {}, {}
        self.lifted, self.placed = set(), set()
        self.closest, self.closing_distance = math.inf, None
        self.previous_grip_max = -math.inf
        self.closing = False
        self.first_grasp, self.first_place = None, None

    def update(self, snapshot):
        import numpy as np
        now = snapshot['sim_ns']/1e9
        if self.last is not None and now <= self.last:
            return
        if self.last is not None and now-self.last > .1:
            raise RuntimeError('Physical scoring snapshots skipped >100ms sim time')
        self.last = now
        if self.begin is None:
            self.begin = now
        target_names = [name for name in self.starts if name.startswith(self.color)]
        if not target_names:
            raise ValueError('Missing requested-colour blocks')
        tip = self.fk.forward_kinematics_batch(np.asarray([snapshot['state'][:5]]))[0]
        # Production FK is robot-base relative; rotate/translate into world.
        from pose_math import rotate
        base = snapshot['robot_base_world_pose']
        tip = np.asarray(rotate(base['orientation'], tip.tolist()))+base['position']
        distance = min(float(np.linalg.norm(tip-np.asarray(snapshot['bodies'][name]['position']))) for name in target_names)
        self.closest = min(self.closest, distance)
        grip = snapshot['state'][5]
        self.previous_grip_max = max(self.previous_grip_max, grip)
        if not self.closing and self.previous_grip_max-grip >= math.radians(2):
            self.closing, self.closing_distance = True, distance
        for name in self.starts:
            height = snapshot['bodies'][name]['position'][2]-self.starts[name]
            if height >= .024:
                self.lift_since.setdefault(name, now)
                if now-self.lift_since[name] >= .5:
                    self.lifted.add(name)
                    if name.startswith(self.color) and self.first_grasp is None:
                        self.first_grasp = self.lift_since[name]-self.begin
            else:
                self.lift_since.pop(name, None)
            for color in COLORS:
                key = (name, color)
                if name in self.lifted and self.contained(snapshot, name, color)['passed']:
                    self.inside_since.setdefault(key, now)
                    if now-self.inside_since[key] >= .5:
                        self.placed.add(key)
                        if name.startswith(self.color) and color == self.color and self.first_place is None:
                            self.first_place = self.inside_since[key]-self.begin
                else:
                    self.inside_since.pop(key, None)
                    self.placed.discard(key)
        return self.result()

    def result(self):
        grasp = any(name.startswith(self.color) for name in self.lifted)
        success = any(name.startswith(self.color) and cup == self.color for name, cup in self.placed)
        return dict(success=success, grasp_success=grasp, correct_cup=success,
                    target_any_placement=any(name.startswith(self.color) for name,cup in self.placed),
                    target_wrong_cup=any(name.startswith(self.color) and cup!=self.color for name,cup in self.placed),
                    wrong_block_placement=any(not name.startswith(self.color) for name,cup in self.placed),
                    any_placement=bool(self.placed), wrong_color_lift=any(not n.startswith(self.color) for n in self.lifted),
                    lifted_objects=sorted(self.lifted), placements=sorted(self.placed),
                    closing_motion_ge2deg=self.closing, closing_distance_m=self.closing_distance,
                    closest_target_distance_m=self.closest if math.isfinite(self.closest) else None,
                    distance_basis='URDF fixed-jaw FK from measured PhysX articulation, transformed by physical base frame',
                    time_to_grasp_s=self.first_grasp, time_to_place_s=self.first_place)
