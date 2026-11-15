"""LeRobot eval robot plugin backed by the qualified collector process."""
from dataclasses import dataclass
from pathlib import Path
from multiprocessing.connection import Client
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

FEATURES=('shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos')

@RobotConfig.register_subclass('collector_native_eval')
@dataclass
class HeldTickRobotConfig(RobotConfig):
    socket: str=''
    auth_file: str=''
    eval_codec_roundtrip: bool=False

class HeldTickRobot(Robot):
    config_class=HeldTickRobotConfig
    name='collector_native_eval'
    def __init__(self,config):
        super().__init__(config);self.config=config;self.connection=None;self.capture_metadata=None;self.codec=None;self.codec_audit_path=None;self.codec_first=True
    @property
    def observation_features(self):return dict(self.action_features,top=(480,640,3),wrist=(480,640,3))
    @property
    def action_features(self):return {name:float for name in FEATURES}
    @property
    def is_connected(self):return self.connection is not None
    @property
    def is_calibrated(self):return True
    def calibrate(self):raise RuntimeError('Native simulator uses the collector articulation contract')
    def configure(self):pass
    def connect(self,calibrate=True):
        if self.connection is not None:raise RuntimeError('Already connected')
        key=Path(self.config.auth_file);assert key.stat().st_mode & 0o077==0
        self.connection=Client(self.config.socket,family='AF_UNIX',authkey=key.read_bytes())
    def exchange(self,payload):
        if self.connection is None:raise RuntimeError('Native transport disconnected')
        self.connection.send(payload)
        if not self.connection.poll(90):raise TimeoutError('Collector eval IPC exceeded90seconds')
        response=self.connection.recv()
        if 'error' in response:raise RuntimeError(response['error'])
        return response
    def get_observation(self):
        response=self.exchange(dict(op='capture'));self.capture_metadata=dict(response['metadata'])
        observation=response['observation']
        if self.config.eval_codec_roundtrip:
            assert self.capture_metadata['mode']=='collect' and self.capture_metadata['held']
            if self.codec is None:raise RuntimeError('Codec stream not reset at episode boundary')
            observation,details=self.codec.transform(observation)
            self.capture_metadata['eval_codec_roundtrip']=details
            if self.codec_first and self.codec_audit_path is not None:
                import numpy as np
                np.savez_compressed(self.codec_audit_path,top=observation['top'],wrist=observation['wrist'])
            self.codec_first=False
        else:self.capture_metadata['eval_codec_roundtrip']=False
        return dict(observation,_eval_native_held=self.capture_metadata)
    def reset_eval_codec(self):
        if self.config.eval_codec_roundtrip:
            import os
            assert os.environ.get('SOARM_EVAL_NATIVE_HELD')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            from codec_stream import EvalCodec
            self.codec=EvalCodec();self.codec_first=True
    def send_action(self,action):
        if set(action)!=set(FEATURES):raise ValueError('Six-joint action contract differs')
        return self.exchange(dict(op='action',action=action))['performed']
    def disconnect(self):
        if self.connection is not None:self.connection.close();self.connection=None
