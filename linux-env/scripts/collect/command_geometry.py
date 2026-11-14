"""Validate exact execution targets and their short swept joint-space segments."""
import math,time


QUERY_POLICY=dict(per_query_timeout_s=1.,delivery_margin_s=.05,max_final_queries=256,
                  max_preflight_queries=50000,max_segment_subdivisions=1000)


def frame_capacity_seconds():
    return QUERY_POLICY["max_final_queries"]*(QUERY_POLICY["per_query_timeout_s"]+QUERY_POLICY["delivery_margin_s"])+57.


class QueryBudget:
    """Native validity RPCs have individual deadlines; successful calls add up."""
    def __init__(self,limit):self.limit=limit;self.count=0;self.total_s=0.;self.maximum_s=0.
    def call(self,check,q):
        if self.count>=self.limit:raise RuntimeError('Execution geometry query-count limit exceeded')
        self.count+=1;start=time.monotonic();answer=check(q);elapsed=time.monotonic()-start
        self.total_s+=elapsed;self.maximum_s=max(self.maximum_s,elapsed)
        if elapsed>QUERY_POLICY['per_query_timeout_s']+QUERY_POLICY['delivery_margin_s']:
            raise TimeoutError('Individual execution geometry query exceeded its deadline')
        if not isinstance(answer,tuple) or len(answer)!=2 or not isinstance(answer[0],bool):
            raise RuntimeError('Execution geometry query returned no valid verdict')
        return answer
    def summary(self):return dict(queries=self.count,total_query_wall_s=self.total_s,
        max_query_wall_s=self.maximum_s,query_limit=self.limit,per_query_timeout_s=QUERY_POLICY['per_query_timeout_s'])


class CommandCollision(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('Execution geometry refused: '+repr(evidence))


def checked_segment(start,target,check,limits,budget=None,spacing=.01):
    """Both endpoints and every <=0.01 rad joint-space subdivision are queried."""
    if len(start)!=6 or len(target)!=6 or not all(math.isfinite(v) for v in [*start,*target]):
        raise ValueError('Incomplete/non-finite execution segment')
    if any(not lo<=q<=hi for q,(lo,hi) in zip(target,limits)):
        return dict(passed=False,reason='joint_limit',target=list(target),checks=0)
    count=max(1,math.ceil(max(abs(a-b) for a,b in zip(start,target))/spacing))
    if count>QUERY_POLICY['max_segment_subdivisions']:raise ValueError('Execution tracking gap too large to validate')
    budget=budget if budget is not None else QueryBudget(count+1)
    if budget.count+count+1>budget.limit:raise RuntimeError('Execution geometry query-count limit exceeded')
    for i in range(count+1):
        u=i/count;q=[a+u*(b-a) for a,b in zip(start,target)]
        valid,contacts=budget.call(check,q)
        if not valid:
            return dict(passed=False,reason='collision',alpha=u,joints=q,checks=i+1,
                contacts=[dict(body_1=c.contact_body_1,body_2=c.contact_body_2,depth=c.depth) for c in contacts[:5]])
    return dict(passed=True,checks=count+1,max_joint_spacing_rad=spacing,target=list(target))


def final_command(candidate,state,previous,check,limits,max_queries=None):
    budget=QueryBudget(QUERY_POLICY['max_final_queries'] if max_queries is None else max_queries)
    checks=[checked_segment(start,candidate.positions,check,limits,budget) for start in (state,previous)]
    result=dict(passed=all(c['passed'] for c in checks),epoch=candidate.epoch,sim_ns=candidate.stamp_ns,
                target=list(candidate.positions),paths=checks,query_timing=budget.summary())
    if not result['passed']:raise CommandCollision(result)
    return result


def choose_noise_scale(schedule,program,gripper,check,limits,require_epoch):
    """Choose a scale for the complete smooth operation before its first step."""
    attempts=[]
    budget=QueryBudget(QUERY_POLICY['max_preflight_queries'])
    for scale in ((1.,.5,.25,0.) if program.segments else (0.,)):
        previous=schedule.sample(0.)+[gripper];failed=None;queries=0
        for tick in range(math.ceil(schedule.duration*30)+1):
            require_epoch()
            t=min(schedule.duration,tick/30)
            q=[a+scale*b for a,b in zip(schedule.sample(t)+[gripper],program.sample(t))]
            result=checked_segment(previous,q,check,limits,budget)
            queries+=result['checks']
            if not result['passed']:failed=result;break
            previous=q
        attempts.append(dict(scale=scale,passed=failed is None,checks=queries,rejection=failed))
        if failed is None:return scale,dict(scale=scale,attempts=attempts,scope='complete arm operation before stepping',query_timing=budget.summary())
    raise CommandCollision(dict(passed=False,phase='noise preflight including zero offset',attempts=attempts))
