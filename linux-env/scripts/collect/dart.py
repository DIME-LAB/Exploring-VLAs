"""Seeded smooth executed perturbations; clean expert labels remain independent."""
from dataclasses import dataclass
import math
import random

DEFAULT_NOISE = dict(version=1, shape='min_jerk_segments', on_probability=.65,
    amplitude_min_rad=math.radians(.26), amplitude_max_rad=math.radians(1.),
    hold_min_s=.3, hold_max_s=.8, ramp_min_s=.35, gap_s=.3,
    noise_velocity_rad_s=.05, noise_acceleration_rad_s2=.30,
    arm_velocity_rad_s=10., arm_acceleration_rad_s2=2., expert_velocity_rad_s=1.,
    expert_acceleration_rad_s2=1.4, quiet_tail_fraction=.20)


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
        if not 0<config['arm_velocity_rad_s']<=10. or not 0<config['arm_acceleration_rad_s2']<=2.:
            raise ValueError('Arm target limits exceed deployment bounds')
        if duration<=0 or any(not math.isfinite(float(v)) for k,v in config.items() if isinstance(v,(int,float))):
            raise ValueError('Invalid smooth noise parameters')
        rng=random.Random(seed);self.segments=[]
        window=duration*(1-config['quiet_tail_fraction'])
        cursor=0.
        if enabled and rng.random()<config['on_probability']:
            while window-cursor>=.6:
                amplitudes=[rng.uniform(config['amplitude_min_rad'],config['amplitude_max_rad'])*rng.choice((-1,1)) for _ in range(5)]
                peak=max(abs(v) for v in amplitudes)
                ramp=max(config['ramp_min_s'],1.875*peak/config['noise_velocity_rad_s'],
                         math.sqrt((10/math.sqrt(3))*peak/config['noise_acceleration_rad_s2']))
                hold=rng.uniform(config['hold_min_s'],config['hold_max_s'])
                available=window-cursor
                if 2*ramp+hold>available:
                    hold=min(hold,max(0.,available/3))
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


class TargetLimiter:
    """Limit arm target derivatives, including discrete stopping room at joint bounds.

    propose is pure with respect to acknowledged state; only commit after the
    physical ACK changes history. Cancellation cannot hand over an unsent target.
    Gripper commands pass through unchanged and have no injected noise.
    """
    def __init__(self, limits, velocity=10., acceleration=2., fps=30):
        self.limits=tuple(limits);self.velocity=velocity;self.acceleration=acceleration;self.dt=1/fps
        self.state=None
        if min(velocity,acceleration,fps)<=0:raise ValueError('Positive command limits required')

    def _safe_speed(self,distance):
        low,high=0.,self.velocity;step=self.acceleration*self.dt
        for _ in range(32):
            v=(low+high)/2;n=math.ceil(v/step)
            stop=self.dt*(n*v-step*n*(n-1)/2)
            if stop<=max(0.,distance):low=v
            else:high=v
        return low

    def propose(self, clean, noise, anchor, epoch, stamp_ns):
        if any(len(v)!=6 or not all(math.isfinite(x) for x in v) for v in (clean,noise,anchor)):
            raise ValueError('Finite six-joint command/anchor required')
        desired=[a+b for a,b in zip(clean,noise)]
        if self.state is None:
            previous=tuple(anchor[:5]);velocities=(0.,)*5
        else:
            if self.state.epoch!=epoch:raise ValueError('Command limiter belongs to a retired epoch')
            previous,velocities=self.state.positions[:5],self.state.velocities
        out=[];next_velocity=[]
        for i,((lo,hi),q,v) in enumerate(zip(self.limits,previous,velocities)):
            if not lo-1e-6<=q<=hi+1e-6:raise ValueError('Physical target anchor outside joint bounds')
            lower=max(-self.velocity,v-self.acceleration*self.dt,-self._safe_speed(q-lo))
            upper=min(self.velocity,v+self.acceleration*self.dt,self._safe_speed(hi-q))
            if lower>upper+1e-8:raise ValueError('No bounded arm target continuation')
            error=desired[i]-q
            feedforward=((desired[i]-self.state.desired_positions[i])/self.dt if self.state is not None else 0.)
            goal_speed=self._safe_speed(abs(error))
            if error>=0:desired_speed=min(error/self.dt,goal_speed+max(0.,feedforward))
            else:desired_speed=max(error/self.dt,-goal_speed+min(0.,feedforward))
            speed=max(lower,min(upper,desired_speed))
            value=q+speed*self.dt
            out.append(value);next_velocity.append(speed)
        out.append(clean[5])
        return TargetCandidate(epoch,stamp_ns,tuple(out),tuple(next_velocity),tuple(noise),
                               tuple(a-b for a,b in zip(out,desired)),tuple(desired))

    def commit(self, candidate, epoch, stamp_ns):
        if candidate.epoch!=epoch or candidate.stamp_ns!=stamp_ns:
            raise ValueError('Stale command limiter candidate')
        self.state=candidate

    def handover(self, epoch):
        if self.state is not None:
            s=self.state
            self.state=TargetCandidate(epoch,s.stamp_ns,s.positions,s.velocities,s.noise_offset,s.filter_delta,s.desired_positions,s.held_reference)


class ExpertSchedule:
    """Retain a bounded planner schedule; otherwise use C2 laws at path corners.

    Source waypoints are never removed or rounded. Only their traversal time
    changes, so the collision-checked joint-space polyline remains identical.
    """
    def __init__(self, trajectory, config):
        self.path=[(p.time_from_start.sec+p.time_from_start.nanosec/1e9,tuple(p.positions)) for p in trajectory.points]
        if (not self.path or self.path[-1][0]<=0 or any(len(q)<5 or not all(math.isfinite(v) for v in q) for _,q in self.path) or
                any(b[0]<a[0] for a,b in zip(self.path,self.path[1:]))):
            raise ValueError('Invalid expert trajectory timing')
        self.source_duration=self.path[-1][0];self.duration=self.source_duration
        self.segments=[];self.original=True
        def peak(sample,duration):
            values=[sample(0.)]*2+[sample(min(duration,i/30)) for i in range(1,math.ceil(duration*30)+1)]
            values+=[values[-1]]*2
            velocity=[[(b[j]-a[j])*30 for j in range(5)] for a,b in zip(values,values[1:])]
            acceleration=[[(b[j]-a[j])*30 for j in range(5)] for a,b in zip(velocity,velocity[1:])]
            return max(abs(v) for row in velocity for v in row),max(abs(v) for row in acceleration for v in row)
        v,a=peak(self._source,self.source_duration)
        if v>config['expert_velocity_rad_s'] or a>config['expert_acceleration_rad_s2']:
            self.original=False
            boundaries=[0];direction=None
            for i in range(1,len(self.path)):
                delta=[b-a for a,b in zip(self.path[i-1][1][:5],self.path[i][1][:5])]
                norm=math.sqrt(sum(d*d for d in delta))
                current=tuple(d/norm for d in delta) if norm>1e-10 else None
                changed=(current is None)!=(direction is None)
                if current is not None and direction is not None:
                    changed=sum(a*b for a,b in zip(current,direction))<1-1e-10
                if changed and i-1>boundaries[-1]:boundaries.append(i-1)
                direction=current
            if boundaries[-1]!=len(self.path)-1:boundaries.append(len(self.path)-1)
            if len(boundaries)==1:boundaries.append(boundaries[0])
            elapsed=0.
            for left,right in zip(boundaries,boundaries[1:]):
                begin,end=self.path[left][0],self.path[right][0]
                source=max(end-begin,1/30)
                delta=max(abs(b-a) for a,b in zip(self.path[left][1][:5],self.path[right][1][:5]))
                duration=max(source,1.875*delta/config['expert_velocity_rad_s'],
                             math.sqrt((10/math.sqrt(3))*delta/config['expert_acceleration_rad_s2']))
                for _ in range(12):
                    def sample(t):return self._source(begin+(end-begin)*min_jerk(t/duration))
                    v,a=peak(sample,duration)
                    factor=max(v/config['expert_velocity_rad_s'],math.sqrt(a/config['expert_acceleration_rad_s2']))
                    if factor<=1+1e-8:break
                    duration*=max(1.02,factor*1.01)
                else:raise ValueError('Cannot retime expert path within target limits')
                self.segments.append(dict(start_s=elapsed,duration_s=duration,source_start_s=begin,source_end_s=end))
                elapsed+=duration
            self.duration=elapsed
        self.spec=dict(source_duration_s=self.source_duration,duration_s=self.duration,
            time_law='planner' if self.original else 'piecewise_minimum_jerk',segments=self.segments,
            velocity_budget_rad_s=config['expert_velocity_rad_s'],acceleration_budget_rad_s2=config['expert_acceleration_rad_s2'])

    def _source(self,t):
        previous=self.path[0]
        for point in self.path[1:]:
            if t<=point[0]:
                alpha=max(0.,min(1.,(t-previous[0])/max(point[0]-previous[0],1e-9)))
                return [a+alpha*(b-a) for a,b in zip(previous[1],point[1])]
            previous=point
        return list(self.path[-1][1])

    def source_time(self,elapsed):
        if self.original:return max(0.,min(self.source_duration,elapsed))
        for segment in self.segments:
            if elapsed<segment['start_s']+segment['duration_s']:
                u=min_jerk((elapsed-segment['start_s'])/segment['duration_s'])
                return segment['source_start_s']+u*(segment['source_end_s']-segment['source_start_s'])
        return self.source_duration

    def sample(self, elapsed):
        return self._source(self.source_time(elapsed))
