"""Opt-in evaluation photometric diagnostic in the isolated sensor process."""
from __future__ import annotations

import numpy as np
import hashlib
import time
from lerobot.async_inference import observation_process as _production

LIVE_GAIN = (1.12162, 1.11828, 1.11650)
ORIGINAL_WORKER = _production._capture_worker


def transform_metadata(enabled):
    return dict(name='inverse_dataset_gain' if enabled else 'identity', enabled=bool(enabled),
                divisor=list(LIVE_GAIN) if enabled else [1., 1., 1.],
                operation='round(clip(rgb / divisor, 0, 255)) -> uint8' if enabled else 'unchanged',
                cameras=['wrist', 'top'])


def inverse_dataset_gain(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Lighting diagnostic requires HxWx3 uint8 RGB images')
    return np.rint(np.clip(rgb.astype(np.float32) / np.asarray(LIVE_GAIN, dtype=np.float32), 0, 255)).astype(np.uint8)


def corrected_capture_worker(*args):
    # Spawn imports this module afresh. Only this sensor process's factory is
    # wrapped; the production worker retains its transport/cancellation logic.
    original_factory = _production.make_robot_from_config
    proof = {}
    connection = args[0]

    class ProofConnection:
        def __getattr__(self, name):
            return getattr(connection, name)
        def send(self, value):
            if proof and isinstance(value, _production.CapturedObservation) and not value.cancelled:
                value.observation_transform_proof = dict(proof)
                proof.clear()
            connection.send(value)

    def corrected_factory(config):
        robot = original_factory(config)
        get_observation = robot.get_observation
        checked = False
        def corrected_observation():
            nonlocal checked
            raw = get_observation()
            for camera in ('wrist', 'top'):
                image = raw[camera]
                corrected = inverse_dataset_gain(image)
                if not checked:
                    proof[camera] = dict(raw_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        corrected_sha256=hashlib.sha256(corrected.tobytes()).hexdigest(),
                        roi_xyxy=[10,260,130,440],roi_kind='table' if camera=='top' else 'fixed_image_roi',
                        raw_roi_median_rgb=np.median(image[260:440,10:130],axis=(0,1)).tolist(),
                        corrected_roi_median_rgb=np.median(corrected[260:440,10:130],axis=(0,1)).tolist(),
                        wall_time=time.time())
                raw[camera] = corrected
            checked = True
            return raw
        robot.get_observation = corrected_observation
        return robot

    _production.make_robot_from_config = corrected_factory
    return ORIGINAL_WORKER(ProofConnection(), *args[1:])


def enable_isolated_correction():
    _production._capture_worker = corrected_capture_worker
