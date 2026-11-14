"""Seeded smooth executed perturbations; clean expert labels remain independent."""
from dataclasses import dataclass
import math
import random

DEFAULT_NOISE = dict(version=2, shape='min_jerk_segments', on_probability=.65,
    amplitude_min_rad=math.radians(.26), amplitude_max_rad=math.radians(1.),
    hold_min_s=.3, hold_max_s=.8, ramp_min_s=.35, gap_s=.3,
    noise_velocity_rad_s=.05, noise_acceleration_rad_s2=.30,
    quiet_tail_fraction=.20)


def min_jerk(u):
    u=max(0.,min(1.,u))
    return u*u*u*(10.+u*(-15.+6.*u))


class SmoothDart:
    def __init__(self, duration, seed, config, enabled=True):
        self.duration,self.seed,self.config=duration,seed,dict(config)
        if config.get('shape')!='min_jerk_segments' or not 0<=config['on_probability']<=1:
            raise ValueError('Unknown smooth noise shape/probability')
        if not 0<=config['amplitude_min_rad']<=config['amplitude_max_rad'] or not 0<config['quiet_tail_fraction']<1:
            raise ValueError('Invalid smooth noise amplitude/window')
        if min(config[k] for k in ('hold_min_s','hold_max_s','ramp_min_s','noise_velocity_rad_s','noise_acceleration_rad_s2'))<=0:
            raise ValueError('Positive smooth ramp and derivative bounds required')
        if duration<=0 or any(not math.isfinite(float(v)) for k,v in config.items() if isinstance(v,(int,float))):
            raise ValueError('Invalid smooth noise parameters')
        rng=random.Random(seed);self.segments=[]
        window=duration*(1-config['quiet_tail_fraction'])
        cursor=0.
        if enabled and rng.random()<config['on_probability']:
            while window-cursor>=2*config['ramp_min_s']+config['hold_min_s']:
                amplitudes=[rng.uniform(config['amplitude_min_rad'],config['amplitude_max_rad'])*rng.choice((-1,1)) for _ in range(5)]
                peak=max(abs(v) for v in amplitudes)
                ramp=max(config['ramp_min_s'],1.875*peak/config['noise_velocity_rad_s'],
                         math.sqrt((10/math.sqrt(3))*peak/config['noise_acceleration_rad_s2']))
                hold=rng.uniform(config['hold_min_s'],config['hold_max_s'])
                available=window-cursor
                if 2*ramp+hold>available:
                    hold=max(config['hold_min_s'],min(hold,available-2*config['ramp_min_s']))
                    ramp=(available-hold)/2
                    if ramp<=0:break
                    scale=min(1.,config['noise_velocity_rad_s']*ramp/(1.875*peak),
                              config['noise_acceleration_rad_s2']*ramp*ramp/((10/math.sqrt(3))*peak))
                    amplitudes=[a*scale for a in amplitudes]
                self.segments.append(dict(start_s=cursor,ramp_s=ramp,hold_s=hold,amplitude_rad=amplitudes))
                cursor+=2*ramp+hold+config['gap_s']
        self.spec=dict(seed=seed,duration_s=duration,enabled=enabled,config=self.config,segments=self.segments)

    def sample(self, seconds):
        if seconds>=self.duration*(1-self.config['quiet_tail_fraction'])-1e-12:
            return [0.]*6
        for segment in self.segments:
            t=seconds-segment['start_s'];r=segment['ramp_s'];h=segment['hold_s']
            if 0<=t<2*r+h:
                weight=min_jerk(t/r) if t<r else 1. if t<r+h else min_jerk((2*r+h-t)/r)
                return [weight*a for a in segment['amplitude_rad']]+[0.]
        return [0.]*6


@dataclass(frozen=True)
class TargetCandidate:
    epoch: int
    stamp_ns: int
    positions: tuple
    velocities: tuple
    noise_offset: tuple
    filter_delta: tuple
    desired_positions: tuple
    held_reference: tuple = None
    planner_reference: tuple = ()
    noise_scale: float = 1.


class PlannerSchedule:
    """Sample the planner's supplied time law without retiming or filtering it."""
    def __init__(self,trajectory,config=None):
        self.path=[(p.time_from_start.sec+p.time_from_start.nanosec/1e9,tuple(p.positions)) for p in trajectory.points]
        if not self.path or self.path[-1][0]<=0 or any(len(q)!=5 or not all(math.isfinite(v) for v in q) for _,q in self.path) or any(b[0]<a[0] for a,b in zip(self.path,self.path[1:])):
            raise ValueError('Invalid planner trajectory')
        self.duration=self.source_duration=self.path[-1][0]
        self.spec=dict(duration_s=self.duration,time_law='planner',waypoints=len(self.path))
    def sample(self,t):
        previous=self.path[0]
        for point in self.path[1:]:
            if t<=point[0]:
                u=max(0.,min(1.,(t-previous[0])/max(point[0]-previous[0],1e-9)))
                return [a+u*(b-a) for a,b in zip(previous[1],point[1])]
            previous=point
        return list(self.path[-1][1])
    def source_time(self,t):return max(0.,min(self.duration,t))
