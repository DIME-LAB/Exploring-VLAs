"""Opt-in, bounded raw wrist history; encode lossless PNG only after execution."""
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import time


class WristHistory:
    def __init__(self, enabled=False, max_actions=1350, window_ns=1_000_000_000):
        if max_actions <= 0 or window_ns <= 0:
            raise ValueError('Positive wrist history limits required')
        self.enabled, self.max_actions, self.window_ns = enabled, max_actions, window_ns
        self.ring, self.selected, self.events = deque(), {}, []
        self.epoch = self.first_ns = self.last_ns = None
        self.last_actions, self.peak_frames = -1, 0
        self.peak_grip, self.minimum_grip, self.latched = -math.inf, math.inf, False
        self.finished = False

    def observe(self, snapshot, image, identity, executed_actions):
        if not self.enabled:
            return
        if self.finished:
            raise ValueError('Wrist history already finished')
        import numpy as np
        stamp, epoch, grip = snapshot['sim_ns'], snapshot['epoch'], snapshot['state'][5]
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError('Wrist retention requires raw HWC uint8 RGB640x480')
        if identity['stream'] != 'wrist' or identity['native_tick']['epoch'] != epoch or abs(identity['native_tick']['sim_ns']-stamp) > 1000 or abs(identity['stamp_ns']-stamp) > 1000:
            raise ValueError('Wrist image does not join the current native tick')
        if not math.isfinite(grip):
            raise ValueError('Nonfinite gripper state')
        if self.last_ns == stamp:
            if epoch != self.epoch or executed_actions != self.last_actions:
                raise ValueError('Duplicate wrist tick changed identity')
            return
        if self.epoch is not None and (epoch != self.epoch or stamp <= self.last_ns):
            raise ValueError('Wrist history crossed an epoch or reversed time')
        if executed_actions != self.last_actions+1 or executed_actions > self.max_actions:
            raise ValueError('Wrist history missed an action tick or exceeded episode limit')
        if self.last_ns is not None and abs(stamp-self.last_ns-1e9/30) > 1000:
            raise ValueError('Wrist history is not one frame per native action tick')
        self.epoch, self.last_ns, self.last_actions = epoch, stamp, executed_actions
        if self.first_ns is None:
            self.first_ns = stamp
        frame = dict(epoch=epoch, sim_ns=stamp, executed_actions=executed_actions,
                     identity=copy.deepcopy(identity), rgb=image.copy())
        self.ring.append(frame)
        while self.ring and self.ring[0]['sim_ns'] < stamp-self.window_ns:
            self.ring.popleft()
        threshold = math.radians(2)
        if self.latched:
            self.minimum_grip = min(self.minimum_grip, grip)
            if grip-self.minimum_grip >= threshold:
                self.latched, self.peak_grip = False, grip
        if not self.latched:
            self.peak_grip = max(self.peak_grip, grip)
            if self.peak_grip-grip >= threshold:
                self.latched, self.minimum_grip = True, grip
                keys = []
                for item in self.ring:
                    key = item['sim_ns'];self.selected[key] = item;keys.append(key)
                self.events.append(dict(epoch=epoch, trigger_sim_ns=stamp,
                                        trigger_executed_actions=executed_actions,
                                        trigger='measured gripper decreased >=2deg since latest opening',
                                        window_ns=self.window_ns, frame_sim_ns=keys,
                                        episode_start_truncated=stamp-self.first_ns < self.window_ns))
        resident = {f['sim_ns'] for f in self.ring} | self.selected.keys()
        self.peak_frames = max(self.peak_frames, len(resident))
        if len(resident) > self.max_actions+1:
            raise ValueError('Wrist history memory bound exceeded')

    def finish(self, directory):
        if not self.enabled:
            return dict(enabled=False)
        if self.finished or self.last_actions != self.max_actions:
            raise ValueError('Wrist history must finish once, after every declared action')
        from PIL import Image
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError('Refusing to overwrite wrist telemetry')
        directory.mkdir(parents=True)
        began = time.monotonic();frames = []
        for stamp, frame in sorted(self.selected.items()):
            filename = f"wrist-{frame['executed_actions']:04d}-{stamp}.png"
            image = frame['rgb'];raw_sha = hashlib.sha256(image.tobytes()).hexdigest()
            path = directory/filename
            Image.fromarray(image).save(path, format='PNG', compress_level=1)
            frames.append(dict(file=filename, epoch=frame['epoch'], sim_ns=stamp,
                               executed_actions=frame['executed_actions'],
                               raw_rgb_sha256=raw_sha, png_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               camera_identity=frame['identity']))
        result = dict(enabled=True, complete=True, format='lossless PNG RGB', window_ns=self.window_ns,
                      verified_native_ticks=self.last_actions+1, executed_actions=self.last_actions,
                      epoch=self.epoch, first_sim_ns=self.first_ns, last_sim_ns=self.last_ns,
                      camera_join_tolerance_ns=1000,
                      capture='existing camera readback after each native ACK, no extra app/physics update',
                      cadence_sim_hz=30, frames=frames, closures=self.events,
                      frames_saved=len(frames), closures_recorded=len(self.events),
                      peak_resident_frames=self.peak_frames,
                      peak_raw_rgb_bytes=self.peak_frames*480*640*3,
                      maximum_raw_rgb_bytes=(self.max_actions+1)*480*640*3,
                      encoding_wall_s=time.monotonic()-began,
                      encoding_phase='after all episode actions; no encoding or I/O in observe')
        manifest = directory/'manifest.json';manifest.write_text(json.dumps(result,indent=2)+'\n')
        summary = dict(enabled=True, complete=True, manifest=str(manifest),
                       verified_native_ticks=result['verified_native_ticks'],
                       executed_actions=result['executed_actions'],
                       epoch=self.epoch, first_sim_ns=self.first_ns, last_sim_ns=self.last_ns,
                       manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                       frames_saved=len(frames), closures_recorded=len(self.events),
                       peak_raw_rgb_bytes=result['peak_raw_rgb_bytes'], encoding_wall_s=result['encoding_wall_s'])
        self.ring.clear();self.selected.clear()
        self.finished = True
        return summary
