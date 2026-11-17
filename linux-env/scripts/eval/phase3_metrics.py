"""Colour-aware physical outcome scoring for Phase3; timestamps are sim seconds."""
from __future__ import annotations

import math
import sys
from pathlib import Path

COLORS = ('red', 'green', 'blue')


class PhysicalScore:
    def __init__(self, color, initial, fk, intended_target=None):
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
        if intended_target is not None and (intended_target not in self.starts or not intended_target.startswith(color+'_')):
            raise ValueError('Intended target must be an existing requested-colour block')
        self.intended_target = intended_target
        self.rule_closest, self.rule_closing_distance = math.inf, None
        self.rule_first_grasp, self.rule_first_place = None, None
        self.rule_closure_confirmed_geometry = None

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
        rule_distance = None
        if self.intended_target is not None:
            rule_distance = float(np.linalg.norm(tip-np.asarray(snapshot['bodies'][self.intended_target]['position'])))
            self.rule_closest = min(self.rule_closest, rule_distance)
        grip = snapshot['state'][5]
        self.previous_grip_max = max(self.previous_grip_max, grip)
        if not self.closing and self.previous_grip_max-grip >= math.radians(2):
            self.closing, self.closing_distance = True, distance
            self.rule_closing_distance = rule_distance
            if self.intended_target is not None:
                from closure_geometry import confirmed_geometry
                self.rule_closure_confirmed_geometry=confirmed_geometry(snapshot,self.intended_target)
        for name in self.starts:
            height = snapshot['bodies'][name]['position'][2]-self.starts[name]
            if height >= .024:
                self.lift_since.setdefault(name, now)
                if now-self.lift_since[name] >= .5:
                    self.lifted.add(name)
                    if name.startswith(self.color) and self.first_grasp is None:
                        self.first_grasp = self.lift_since[name]-self.begin
                    if name == self.intended_target and self.rule_first_grasp is None:
                        self.rule_first_grasp = self.lift_since[name]-self.begin
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
                        if name == self.intended_target and color == self.color and self.rule_first_place is None:
                            self.rule_first_place = self.inside_since[key]-self.begin
                else:
                    self.inside_since.pop(key, None)
                    self.placed.discard(key)
        return self.result()

    def result(self):
        grasp = any(name.startswith(self.color) for name in self.lifted)
        success = any(name.startswith(self.color) and cup == self.color for name, cup in self.placed)
        result = dict(success=success, grasp_success=grasp, correct_cup=success,
                    target_any_placement=any(name.startswith(self.color) for name,cup in self.placed),
                    target_wrong_cup=any(name.startswith(self.color) and cup!=self.color for name,cup in self.placed),
                    wrong_block_placement=any(not name.startswith(self.color) for name,cup in self.placed),
                    any_placement=bool(self.placed), wrong_color_lift=any(not n.startswith(self.color) for n in self.lifted),
                    lifted_objects=sorted(self.lifted), placements=sorted(self.placed),
                    closing_motion_ge2deg=self.closing, closing_distance_m=self.closing_distance,
                    closest_target_distance_m=self.closest if math.isfinite(self.closest) else None,
                    distance_basis='URDF fixed-jaw FK from measured PhysX articulation, transformed by physical base frame',
                    time_to_grasp_s=self.first_grasp, time_to_place_s=self.first_place)
        if self.intended_target is not None:
            result.update(requested_colour_success=success, requested_colour_grasp=grasp,
                          rule_target_name=self.intended_target,
                          rule_target_success=(self.intended_target,self.color) in self.placed,
                          rule_target_grasp=self.intended_target in self.lifted,
                          wrong_same_colour_target_grasp=any(n.startswith(self.color+'_') and n!=self.intended_target for n in self.lifted),
                          wrong_same_colour_target_placement=any(n.startswith(self.color+'_') and n!=self.intended_target for n,c in self.placed),
                          rule_closing_distance_m=self.rule_closing_distance,
                          rule_closest_target_distance_m=self.rule_closest if math.isfinite(self.rule_closest) else None,
                          rule_closure_confirmed_geometry=self.rule_closure_confirmed_geometry,
                          time_to_rule_grasp_s=self.rule_first_grasp,time_to_rule_place_s=self.rule_first_place)
        return result
