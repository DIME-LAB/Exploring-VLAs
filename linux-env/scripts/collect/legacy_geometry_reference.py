"""Frozen pure geometry AST from verified GUI c3da8b95c49c4538ce741d2f2deca663fe1b31cf; parity oracle only."""
import math
JOINT_LIMITS = {'shoulder_pan': (-1.91986, 1.91986), 'shoulder_lift': (-1.74533, 1.74533), 'elbow_flex': (-1.69, 1.69), 'wrist_flex': (-1.65806, 1.65806), 'wrist_roll': (-2.74385, 2.84121), 'gripper_joint': (-0.174533, 1.74533)}
BASELINE_JAW_GAP = 0.019
JAW_GAP_RATE = 0.0749
JAW_OPEN_CLEARANCE_M = 0.005
JAW_CLOSE_CLEARANCE_M = 0.0
TCP_CLEARANCE_M = 0.001
_WRIST_ROLL_URDF_PITCH = 0.0487
_CHAIN_BASE_ROTATION = math.pi / 2 - _WRIST_ROLL_URDF_PITCH
_GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M = 0.00165

def _normalize_grasp_yaw(yaw, pan, current_wrist_roll=None):
    """Pick yaw or yaw±π that minimizes wrist_roll motion from the current pose.

    Gripper jaws are symmetric about the grip axis, so yaw and yaw+π
    produce equivalent grasps. Among the in-limit candidates we pick
    whichever puts the target wrist_roll closest to ``current_wrist_roll`` —
    this minimizes the swing the arm has to perform to reach the grasp.
    Falls back to the joint-range center as the reference when
    ``current_wrist_roll`` is None (preserves deterministic behavior for
    callers that don't track joint state).
    """
    (wr_lo, wr_hi) = JOINT_LIMITS['wrist_roll']
    ref = current_wrist_roll if current_wrist_roll is not None else (wr_lo + wr_hi) / 2
    best = None
    best_dist = float('inf')
    for candidate in (yaw, yaw + math.pi, yaw - math.pi):
        target_wr = pan + candidate - _CHAIN_BASE_ROTATION
        if not wr_lo <= target_wr <= wr_hi:
            continue
        dist = abs(target_wr - ref)
        if dist < best_dist:
            (best, best_dist) = (candidate, dist)
    if best is None:
        wr_center = (wr_lo + wr_hi) / 2
        (best, best_dist) = (yaw, abs(pan + yaw - _CHAIN_BASE_ROTATION - wr_center))
        for candidate in (yaw + math.pi, yaw - math.pi):
            dist = abs(pan + candidate - _CHAIN_BASE_ROTATION - wr_center)
            if dist < best_dist:
                (best, best_dist) = (candidate, dist)
    return best

class LegacyGeometry:

    def _get_grip_width(self, obj_name):
        """Return the grip width for the object, respecting cross-axis checkbox."""
        bbox = self._lookup_bbox(obj_name)
        if not bbox:
            return None
        cross = self._grasp_cross_var.get() if hasattr(self, '_grasp_cross_var') else False
        return max(bbox['sx'], bbox['sy']) if cross else min(bbox['sx'], bbox['sy'])

    def _compute_jaw_offset(self, obj_name, obj_yaw):
        """Compute (dx, dy) to shift TCP so the object center sits between jaws.

        TCP is at the fixed jaw tip. The moving jaw opens along the object's
        local +Y axis: (-sin(obj_yaw), cos(obj_yaw)) in world frame.
        (wrist_roll compensates for pan, so only obj_yaw matters.)
        We shift TCP from object center toward the fixed jaw by grip_width/2.
        Returns (0, 0) when TCP offset is disabled via the checkbox.
        """
        if hasattr(self, '_bbox_enabled_var') and (not self._bbox_enabled_var.get()):
            return (0.0, 0.0)
        grip_width = self._get_grip_width(obj_name)
        if grip_width is None:
            return (0.0, 0.0)
        tcp_clear = self._tcp_clearance_var.get() / 1000 if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M
        half_offset = grip_width / 2 + tcp_clear
        dx = -half_offset * math.sin(obj_yaw)
        dy = half_offset * math.cos(obj_yaw)
        return (dx, dy)

    def _gripper_angle_for_object(self, obj_name):
        """Return (open_angle, close_angle) in radians for the selected object.

        Uses jaw_gap = BASELINE_JAW_GAP + JAW_GAP_RATE * angle model derived
        from STL mesh analysis. Falls back to full range if no bbox data.
        """
        grip_width = self._get_grip_width(obj_name) if obj_name else None
        if grip_width is None:
            return (JOINT_LIMITS['gripper_joint'][1], JOINT_LIMITS['gripper_joint'][0])
        tcp_clear = self._tcp_clearance_var.get() / 1000 if hasattr(self, '_tcp_clearance_var') else TCP_CLEARANCE_M
        open_cl = self._jaw_open_clearance_var.get() / 1000 if hasattr(self, '_jaw_open_clearance_var') else JAW_OPEN_CLEARANCE_M
        close_cl = self._jaw_close_clearance_var.get() / 1000 if hasattr(self, '_jaw_close_clearance_var') else JAW_CLOSE_CLEARANCE_M
        open_gap = grip_width + 2 * tcp_clear + open_cl
        open_angle = (open_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        open_angle = max(JOINT_LIMITS['gripper_joint'][0], min(open_angle, JOINT_LIMITS['gripper_joint'][1]))
        close_gap = grip_width + 2 * tcp_clear + close_cl
        close_angle = (close_gap - BASELINE_JAW_GAP) / JAW_GAP_RATE
        close_angle = max(JOINT_LIMITS['gripper_joint'][0], min(close_angle, JOINT_LIMITS['gripper_joint'][1]))
        return (open_angle, close_angle)

    def plan(self, pos):
        obj_name = 'block'
        topic = self._grasp_topic_var.get().strip()
        z_offset = 0.05 if topic == '/drop_poses' else 0.0
        obj_z_override = self._grasp_obj_z_var.get()
        if abs(obj_z_override) > 0.0001:
            base_z = obj_z_override
        else:
            bbox_sz = (self._lookup_bbox(obj_name) or {}).get('sz', 0.0)
            base_z = pos['z'] + float(bbox_sz) / 2.0 + _GRIPPER_TCP_CLEARANCE_ABOVE_BLOCK_M
        target_z = base_z + z_offset
        action = 'drop' if topic == '/drop_poses' else 'grab'
        obj_qz = pos.get('qz', 0.0)
        obj_qw = pos.get('qw', 1.0)
        obj_yaw = math.atan2(2.0 * obj_qw * obj_qz, 1.0 - 2.0 * obj_qz * obj_qz)
        cross = self._grasp_cross_var.get()
        if cross:
            obj_yaw += math.pi / 2
        pan = math.atan2(pos['y'], pos['x']) if abs(pos['x']) + abs(pos['y']) > 0.001 else 0.0
        with self.joint_lock:
            current_wr = self._actual_positions.get('wrist_roll')
        obj_yaw = _normalize_grasp_yaw(obj_yaw, pan, current_wrist_roll=current_wr)
        (jaw_dx, jaw_dy) = self._compute_jaw_offset(obj_name, obj_yaw)
        (tx, ty) = (pos['x'] + jaw_dx, pos['y'] + jaw_dy)
        approach_h = self._grasp_approach_height_var.get()
        poses_to_check = []
        if approach_h > 0:
            approach_z = target_z + approach_h
            poses_to_check.append(('approach', tx, ty, approach_z))
            poses_to_check.append(('final', tx, ty, target_z))
        else:
            poses_to_check.append(('final', tx, ty, target_z))
        return dict(tcp=(tx, ty, target_z), yaw=obj_yaw, jaw_offset=(jaw_dx, jaw_dy), poses=poses_to_check)
