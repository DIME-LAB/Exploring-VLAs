"""Validate exact execution targets and their short swept joint-space segments."""
import math,time


class CommandCollision(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('Execution geometry refused: '+repr(evidence))


def checked_segment(start,target,check,limits,deadline,spacing=.01):
    """Both endpoints and every <=0.01 rad joint-space subdivision are queried."""
    if len(start)!=6 or len(target)!=6 or not all(math.isfinite(v) for v in [*start,*target]):
        raise ValueError('Incomplete/non-finite execution segment')
    if any(not lo<=q<=hi for q,(lo,hi) in zip(target,limits)):
        return dict(passed=False,reason='joint_limit',target=list(target),checks=0)
    count=max(1,math.ceil(max(abs(a-b) for a,b in zip(start,target))/spacing))
    if count>1000:raise ValueError('Execution tracking gap too large to validate')
    for i in range(count+1):
        if time.monotonic()>deadline:raise TimeoutError('Final execution geometry query budget exceeded')
        u=i/count;q=[a+u*(b-a) for a,b in zip(start,target)]
        valid,contacts=check(q)
        if not valid:
            return dict(passed=False,reason='collision',alpha=u,joints=q,checks=i+1,
                contacts=[dict(body_1=c.contact_body_1,body_2=c.contact_body_2,depth=c.depth) for c in contacts[:5]])
    return dict(passed=True,checks=count+1,max_joint_spacing_rad=spacing,target=list(target))


def final_command(candidate,state,previous,check,limits,budget_s=5.):
    deadline=time.monotonic()+budget_s
    checks=[checked_segment(start,candidate.positions,check,limits,deadline) for start in (state,previous)]
    result=dict(passed=all(c['passed'] for c in checks),epoch=candidate.epoch,sim_ns=candidate.stamp_ns,
                target=list(candidate.positions),paths=checks)
    if not result['passed']:raise CommandCollision(result)
    return result


def choose_noise_scale(schedule,program,gripper,check,limits,require_epoch):
    """Choose a scale for the complete smooth operation before its first step."""
    attempts=[]
    for scale in ((1.,.5,.25,0.) if program.segments else (0.,)):
        previous=schedule.sample(0.)+[gripper];failed=None;queries=0
        deadline=time.monotonic()+120.
        for tick in range(math.ceil(schedule.duration*30)+1):
            require_epoch()
            t=min(schedule.duration,tick/30)
            q=[a+scale*b for a,b in zip(schedule.sample(t)+[gripper],program.sample(t))]
            result=checked_segment(previous,q,check,limits,deadline)
            queries+=result['checks']
            if not result['passed']:failed=result;break
            previous=q
        attempts.append(dict(scale=scale,passed=failed is None,checks=queries,rejection=failed))
        if failed is None:return scale,dict(scale=scale,attempts=attempts,scope='complete arm operation before stepping')
    raise CommandCollision(dict(passed=False,phase='noise preflight including zero offset',attempts=attempts))
