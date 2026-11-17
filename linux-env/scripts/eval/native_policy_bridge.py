"""Native IPC observation/action handlers shared by the live driver and CPU tests."""
import hashlib
import math
import numpy as np


def capture_observation(transport, features, current, first_images, wrist_history, executed_actions):
    observation={name:math.degrees(q) for name,q in zip(features,current['state'])}
    images={cam:transport.sensors.image(transport.session.entry[cam][0]).copy() for cam in ('top','wrist')}
    observation.update(images)
    if not first_images:first_images.update(images)
    if wrist_history is not None:wrist_history.observe(current,images['wrist'],transport.session.entry['wrist'][2],executed_actions)
    metadata=dict(version=1,mode='collect',held=True,sim_ns=current['sim_ns'],epoch=current['epoch'],
                  state_degrees=[observation[n] for n in features],
                  camera_rgb_sha256={cam:hashlib.sha256(image.tobytes()).hexdigest() for cam,image in images.items()},
                  camera_identity={cam:transport.session.entry[cam][2] for cam in images})
    return observation,metadata


def apply_action(transport,features,bounds,hold,warm,action,commands,snapshots,scorer,wrist_history):
    values=np.asarray([action[name] for name in features])
    if values.shape!=(6,) or not np.all(np.isfinite(values)):raise ValueError('Native action must contain six finite degree targets')
    command=hold if warm else np.clip(np.radians(values),bounds[:,0],bounds[:,1]).tolist()
    proof,current=transport.session.step(command)
    commands.append(dict(action_index=len(commands),requested_degrees=values.tolist(),warmup_hold=warm,**proof))
    snapshots.append(current);scorer.update(current)
    if wrist_history is not None:wrist_history.observe(current,transport.sensors.image(transport.session.entry['wrist'][0]),transport.session.entry['wrist'][2],len(commands))
    return dict(performed={name:math.degrees(q) for name,q in zip(features,proof['ack']['applied_command'])})
