"""Agreement rules distinguish copied identities from independent numeric producers."""
import math

# These preserve the capture/ACK contract bounds. No relative tolerance widens
# them at large timestamps or joint angles. Units are part of each rule's name.
TOLERANCES=dict(copy=0.,stamp_ns=1000.,native_ns=10.,state_rad=1e-7,
                target_rad=1e-6,arithmetic=1e-9,velocity_rad_s=1e-4,seed_math=1e-12,
                world_s=1e-6,pose_m=1e-5,float32_readback=1e-6)


def comparison(field,left,right,rule='copy',*,tolerance=None):
    limit=TOLERANCES[rule] if tolerance is None else tolerance
    if rule=='copy':
        passed=left==right;delta=None
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            delta=left-right;passed=passed and math.isfinite(left) and math.isfinite(right)
    else:
        def differences(a,b):
            if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
                if len(a)!=len(b):raise ValueError('Different vector lengths')
                return [v for x,y in zip(a,b) for v in differences(x,y)]
            if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('Missing or non-finite numeric producer')
            return [a-b]
        try:
            values=differences(left,right);delta=max(values,key=abs) if values else 0.
            passed=abs(delta)<=limit
        except (TypeError,ValueError):delta=None;passed=False
    return dict(field=field,left=left,right=right,delta=delta,rule=rule,tolerance=limit,passed=passed)


def matches(left,right,rule='copy',**kwargs):return comparison('',left,right,rule,**kwargs)['passed']


def bound_comparison(field,value,lower,upper,rule):
    valid=isinstance(value,(int,float)) and math.isfinite(value)
    delta=(value-lower if value<lower else value-upper if value>upper else 0.) if valid else None
    return dict(field=field,left=value,right=[lower,upper],delta=delta,rule=rule,
                tolerance=TOLERANCES[rule],passed=valid and abs(delta)<=TOLERANCES[rule])


def interval(value,lower,upper,rule):
    """A declared latency/window bound; integer order stays strict elsewhere."""
    return bound_comparison('',value,lower,upper,rule)['passed']


def native_frame(row):
    # These fields were copied from the same native snapshot, unlike the
    # camera-anchored row stamp and the separately published measured JointState.
    return dict(epoch=row['epoch'],sim_ns=row.get('pose_stamp_ns',row['sim_ns']),
        state_source=row['state_source'],state=row.get('physics_state',row['state']),bodies=row['physical_bodies'])
