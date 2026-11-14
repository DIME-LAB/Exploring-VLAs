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


class ClearanceDart:
    """Smooth drift inside the open-jaw corridor; quiet only at final contact.

    Conservative joint-pivot-to-pad radii bound every pad-point displacement.
    This changes the noise offset only; the planner time law stays untouched.
    """
    POLICY=dict(version=1,near_budget_max_m=.0002,corridor_fraction=.2,
        above_block_fraction=.4,contact_fade_m=.003,pad_below_tcp_m=.0081,
        lever_bounds_m=[.50,.50,.35,.25,.15],derivative_check_hz=120)
    def __init__(self,base,schedule,fk,block_top_m,final_tcp_z_m,tcp_clearance_m):
        self.base,self.schedule,self.fk=base,schedule,fk
        self.duration,self.segments=base.duration,base.segments
        self.block_top,self.final_z=block_top_m,final_tcp_z_m
        self.near_budget=min(self.POLICY['near_budget_max_m'],max(0.,tcp_clearance_m)*self.POLICY['corridor_fraction'])
        self.derivative_scale=1.
        hz=self.POLICY['derivative_check_hz'];values=[self.raw(i/hz) for i in range(math.ceil(self.duration*hz)+1)]
        velocities=[[(b[j]-a[j])*hz for j in range(5)] for a,b in zip(values,values[1:])]
        accelerations=[[(b[j]-a[j])*hz for j in range(5)] for a,b in zip(velocities,velocities[1:])]
        vmax=max((abs(v) for q in velocities for v in q),default=0.);amax=max((abs(v) for q in accelerations for v in q),default=0.)
        self.derivative_scale=min(1.,base.config['noise_velocity_rad_s']/max(vmax,1e-12),base.config['noise_acceleration_rad_s2']/max(amax,1e-12))
        self.spec=dict(base.spec,clearance_policy=self.POLICY,near_budget_m=self.near_budget,
            block_top_m=block_top_m,final_tcp_z_m=final_tcp_z_m,derivative_scale=self.derivative_scale,
            raw_noise_velocity_max_rad_s=vmax,raw_noise_acceleration_max_rad_s2=amax)
    def height(self,t):return float(self.fk(self.schedule.sample(t))[2])
    def raw(self,t):
        noise=self.base.sample(t);z=self.height(t)
        above=z-self.POLICY['pad_below_tcp_m']-self.block_top
        # Smooth positive-part avoids a kink at the block's upper surface.
        positive=.5*(above+math.sqrt(above*above+1e-12))
        budget=self.near_budget+self.POLICY['above_block_fraction']*positive
        norm=math.sqrt(5*sum((r*q)**2 for r,q in zip(self.POLICY['lever_bounds_m'],noise[:5])))
        gain=1./math.sqrt(1.+(norm/max(budget,1e-15))**2)
        gain*=min_jerk((z-self.final_z)/self.POLICY['contact_fade_m'])
        return [v*gain for v in noise[:5]]+[0.]
    def sample(self,t):return [self.derivative_scale*v for v in self.raw(t)]
    def phase_statistics(self,scale=1.):
        result={}
        for tick in range(math.ceil(self.duration*30)+1):
            t=min(self.duration,tick/30);z=self.height(t)
            phase='contact' if z-self.final_z<=self.POLICY['contact_fade_m'] else 'near_block' if z-self.POLICY['pad_below_tcp_m']-self.block_top<=.010 else 'high_approach'
            q=[v*scale for v in self.sample(t)];s=result.setdefault(phase,dict(ticks=0,nonzero_ticks=0,max_joint_offset_rad=0.,max_pad_displacement_bound_m=0.))
            s['ticks']+=1;s['nonzero_ticks']+=int(any(abs(v)>1e-12 for v in q))
            s['max_joint_offset_rad']=max(s['max_joint_offset_rad'],max(abs(v) for v in q))
            s['max_pad_displacement_bound_m']=max(s['max_pad_displacement_bound_m'],sum(r*abs(v) for r,v in zip(self.POLICY['lever_bounds_m'],q)))
        return result


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
