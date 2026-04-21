"""Forensic tracing for SO-ARM101 grasp / drop / home cycles.

One JSON file per pick-place cycle. Scope runs from `_cmd_grasp_move`
(open_cycle) to `_cmd_grasp_home` (close_cycle). All events between
those two calls are recorded in the same file.

Events carry a monotonic timestamp so timing histograms fall out for
free (grasp_move_start -> gate_a_done -> gate_b_done -> ... ->
execute_done -> drop_sweep_start -> ...).

Scene snapshots are triggered at boundaries where staleness matters
(before_ompl_plan, after_ompl_returns, after_waypoint_check) so we
can diff them post-hoc to prove / disprove the "scene changed between
plan and check" hypothesis for Mode B failures.

Output directory:
    /home/aaugus11/Documents/isaac-sim-mcp/.planning/phases/
    09-collision-scene-completeness/traces/cycle_{ts}.json

Hot-reload safe: state lives on the singleton which is re-created on
module reimport; open cycles are dropped on reload (by design — a
hot-reload mid-cycle invalidates the trace anyway).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

TRACE_DIR = Path(
    '/home/aaugus11/Documents/isaac-sim-mcp/.planning/phases/'
    '09-collision-scene-completeness/traces'
)


def _json_safe(value: Any) -> Any:
    """Make values JSON-serialisable (numpy scalars, sets, bytes, etc.)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_json_safe(v) for v in value]
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return repr(value)
    for attr in ('tolist', 'item'):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn())
            except Exception:
                pass
    return repr(value)


class Tracer:
    """Singleton trace recorder. Call module-level `tracer` below."""

    def __init__(self, trace_dir: Path = TRACE_DIR) -> None:
        self._dir = trace_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cycle: Optional[dict] = None
        self._t0: float = 0.0

    # ------------------------------------------------------------------
    # Cycle lifecycle
    # ------------------------------------------------------------------

    def open_cycle(self, **meta: Any) -> None:
        """Begin a new cycle. If one is already open it is force-closed
        with outcome='abandoned' so we never silently lose a trace."""
        with self._lock:
            if self._cycle is not None:
                self._close_locked(outcome='abandoned',
                                   note='new cycle opened before previous closed')
            self._t0 = time.monotonic()
            self._cycle = {
                'opened_wall': time.time(),
                'opened_iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'meta': _json_safe(meta),
                'events': [],
            }

    def close_cycle(self, outcome: str, **data: Any) -> Optional[Path]:
        with self._lock:
            return self._close_locked(outcome=outcome, **data)

    def _close_locked(self, outcome: str, **data: Any) -> Optional[Path]:
        if self._cycle is None:
            return None
        self._cycle['outcome'] = outcome
        self._cycle['outcome_data'] = _json_safe(data)
        self._cycle['closed_wall'] = time.time()
        self._cycle['duration_s'] = round(time.monotonic() - self._t0, 4)
        fname = time.strftime('cycle_%Y%m%d_%H%M%S') + '.json'
        path = self._dir / fname
        # If file exists (sub-second collision), disambiguate.
        n = 1
        while path.exists():
            path = self._dir / (fname[:-5] + f'_{n}.json')
            n += 1
        try:
            path.write_text(json.dumps(self._cycle, indent=2))
        except Exception as e:
            # Tracing must not break the robot. Log to stderr and drop.
            print(f'[grasp_trace] Failed to write {path}: {e}')
            path = None
        self._cycle = None
        return path

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def event(self, stage: str, **data: Any) -> None:
        """Record a stage event. No-op if no cycle is open (keeps call
        sites safe when tracing isn't started)."""
        with self._lock:
            if self._cycle is None:
                return
            self._cycle['events'].append({
                't': round(time.monotonic() - self._t0, 4),
                'stage': stage,
                'data': _json_safe(data),
            })

    def is_active(self) -> bool:
        with self._lock:
            return self._cycle is not None

    # ------------------------------------------------------------------
    # Scene snapshotting helpers
    # ------------------------------------------------------------------

    def snapshot_scene(self, label: str, gui: Any) -> None:
        """Capture planning-scene-relevant state from the GUI at a
        named boundary. Failures here are soft — a missing snapshot
        must not corrupt the cycle or break the robot.

        Captured:
          - objects_data (lego positions)
          - objects_bbox (lego dims)
          - attached_lego_name
          - lego_collision_names (which are in the world scene)
          - current joint positions
        """
        try:
            with getattr(gui, 'objects_lock', _NullLock()):
                objects_data = dict(getattr(gui, 'objects_data', {}) or {})
                objects_bbox = dict(getattr(gui, 'objects_bbox', {}) or {})
            with getattr(gui, 'joint_lock', _NullLock()):
                actual = dict(getattr(gui, '_actual_positions', {}) or {})
            snap = {
                'attached_lego': getattr(gui, '_attached_lego_name', None),
                'lego_collision_names':
                    list(getattr(gui, '_lego_collision_names', []) or []),
                'objects': objects_data,
                'objects_bbox': objects_bbox,
                'actual_joints': actual,
            }
            self.event(f'scene_snapshot:{label}', scene=snap)
        except Exception as e:
            self.event(f'scene_snapshot_error:{label}', error=repr(e))


class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# Module-level singleton. `from .grasp_trace import tracer`
tracer = Tracer()
