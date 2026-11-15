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

class HeldTickRobot(Robot):
    config_class=HeldTickRobotConfig
    name='collector_native_eval'
    def __init__(self,config):
        super().__init__(config);self.config=config;self.connection=None;self.capture_metadata=None
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
        response=self.exchange(dict(op='capture'));self.capture_metadata=response['metadata']
        return dict(response['observation'],_eval_native_held=response['metadata'])
    def send_action(self,action):
        if set(action)!=set(FEATURES):raise ValueError('Six-joint action contract differs')
        return self.exchange(dict(op='action',action=action))['performed']
    def disconnect(self):
        if self.connection is not None:self.connection.close();self.connection=None
