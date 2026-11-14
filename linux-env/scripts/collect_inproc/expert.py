"""Sequential scripted expert using the shared SO-ARM101 geometry and time law."""
from types import SimpleNamespace as NS
import math
import numpy as np
import shared
from contract import JOINTS
from dart import PlannerSchedule, SmoothDart
from command_geometry import checked_segment, choose_noise_scale, CommandCollision
from episode_policy import QualityReject, ConvergenceFailure
from recovery import recovery_geometry, detect_miss
from comparisons import native_frame
from scene_sync import physical_pose_caches
from so_arm101_control.grasp_geometry import (make_grasp_plan,find_reachable_grasp_yaw,
    gripper_angles,_grasp_home_data_target,JOINT_LIMITS,BASELINE_JAW_GAP,JAW_GAP_RATE)
from so_arm101_control.compute_workspace import geometric_ik, X_PAN
from tick_contract import digest, require_snapshots


def schedule(start,end,duration):
    """The shared direct joint planner's original linear time law."""
    points=[]
    for index in range(51):
        t=index*duration/50;sec=int(t)
        points.append(NS(positions=[a+index/50*(b-a) for a,b in zip(start,end)],
            time_from_start=NS(sec=sec,nanosec=round((t-sec)*1e9))))
    return PlannerSchedule(NS(points=points))


class Expert:
    def __init__(self,backend,geometry,tick,params,episode_id):
        self.backend,self.geometry,self.tick=backend,geometry,tick
        self.params,self.episode_id=params,episode_id
        self.reference=list(backend.snapshot()['state'])
        self.executed_hold=list(self.reference)
        self.operation=0;self.noise_specs=[];self.rows=[];self.snapshots=[]
        self.limits=[JOINT_LIMITS[n] for n in JOINTS]

    def emit(self,reference,executed,noise,phase):
        row,snapshot=self.tick(reference,executed,noise,phase)
        self.rows.append(row);self.snapshots.append(snapshot)
        return snapshot

    def select(self,solutions):
        self.geometry.update(self.backend.snapshot())
        for solution in solutions:
            q=[solution[n] for n in JOINTS[:5]]+[self.reference[5]]
            if self.geometry.check(q)[0]:return q[:5]
        self.reject('planner_reject',dict(reason='no collision-free shared IK solution'))

    def reject(self,kind,evidence,window=None):
        window=self.snapshots[-2:] if window is None else window
        contract=require_snapshots(window)
        raise QualityReject(kind,evidence,contract)

    def arm(self,target,phase,duration=3.,executed_target=None,hold=False,noise=True):
        start=self.backend.snapshot();q0=start['state'][:5]
        clean=schedule(q0,target,duration/self.params['speed_scale'])
        executed=schedule(q0,target if executed_target is None else executed_target,clean.duration)
        self.geometry.update(start)
        # Both label and execution plans are checked before the first target is applied.
        try:
            for path in (clean,executed):
                previous=q0+[self.reference[5]]
                for i in range(51):
                    q=path.sample(i*path.duration/50)+[self.reference[5]]
                    result=checked_segment(previous,q,self.geometry.check,self.limits)
                    if not result['passed']:raise CommandCollision(result)
                    previous=q
            seed=self.params['noise_seed']+self.operation*104729
            enabled=noise and phase not in ('grasp_home','grasp_home_data','home') and not phase.startswith('regrasp_miss')
            program=SmoothDart(executed.duration,seed,self.params['dart'],enabled=enabled)
            scale,proof=choose_noise_scale(executed,program,self.reference[5],self.geometry.check,
                self.limits,lambda: self.require_epoch(start['epoch']))
        except CommandCollision as exc:self.reject('planner_reject',exc.evidence)
        self.noise_specs.append(dict(phase=phase,scale=scale,preflight=proof,**program.spec))
        self.operation+=1;first=self.backend.stamp()
        for _ in range(math.ceil((clean.duration+5)*30)+2):
            s=self.backend.snapshot();elapsed=(s['sim_ns']-first)/1e9
            self.reference[:5]=clean.sample(elapsed+1/30)
            executed_q=executed.sample(elapsed+1/30)+[self.reference[5]]
            self.emit(list(self.reference),executed_q,[v*scale for v in program.sample(elapsed+1/30)],phase)
            if elapsed>clean.duration and max(abs(a-b) for a,b in zip(s['state'][:5],executed_q[:5]))<.008:
                self.executed_hold=executed_q if hold else list(self.reference)
                return
            if elapsed>clean.duration+5:
                self.reject('arm_nonconvergence',dict(phase=phase,elapsed_s=elapsed,target=executed_q))
        raise RuntimeError('Arm loop exhausted its tick budget')

    def require_epoch(self,epoch):
        if self.backend.epoch!=epoch:raise RuntimeError('Plan belongs to an earlier reset')

    def grip(self,target,phase,duration=3.):
        initial=self.backend.snapshot()['state'][5];first=self.backend.stamp()
        for _ in range(math.ceil((duration+5)*30)+2):
            s=self.backend.snapshot();elapsed=(s['sim_ns']-first)/1e9
            self.reference[5]=initial+min(1.,(elapsed+1/30)/duration)*(target-initial)
            self.emit(list(self.reference),list(self.executed_hold),[0.]*6,phase)
            if elapsed>duration and abs(s['state'][5]-target)<.008:return
            if elapsed>duration+5:self.reject('gripper_nonconvergence',dict(phase=phase,target=target,measured=s['state'][5]))
        raise RuntimeError('Gripper loop exhausted its tick budget')

    def current_geometry(self):
        s=self.backend.snapshot();objects,cups=physical_pose_caches(s,primitive_cups=True)
        name=self.params['object'];corners=np.asarray(s['local_corners'][name])
        bbox=dict(zip(('sx','sy','sz'),np.ptp(corners,axis=0)))
        return s,objects[name],bbox,cups

    def recover(self):
        s,pose,bbox,_=self.current_geometry()
        geometry=recovery_geometry(pose,bbox,self.params['regrasp'],
            current_wrist_roll=s['state'][4],approach_height=self.params['approach_height_m'],
            approach_offset=self.params['approach_offset_m'])
        report=dict(epoch=s['epoch'],episode_id=self.episode_id,kind='regrasp',
            geometry=geometry,miss_confirmed=False,steps=[])
        if geometry['yaw'] is None:self.reject('planner_reject',dict(reason='regrasp IK unreachable'))
        solutions={name:self.select(v) for name,v in geometry['solutions'].items()}
        opening,closing=gripper_angles(min(bbox['sx'],bbox['sy']))
        def grip(phase,q):
            self.grip(q,phase)
            report['steps'].append(dict(stage=phase,sim_ns=self.snapshots[-1]['sim_ns']))
        def arm(phase,clean_key,executed_key):
            self.arm(solutions[clean_key],phase,2.5,executed_target=solutions[executed_key],hold=True)
            report['steps'].append(dict(stage=phase,end_sim_ns=self.snapshots[-1]['sim_ns']))
        grip('regrasp_miss_open',opening)
        arm('regrasp_miss_approach','true_approach','miss_approach')
        arm('regrasp_miss_descent','true_final','miss_final')
        before=self.snapshots[-1]['sim_ns']
        grip('regrasp_miss_close',closing)
        arm('regrasp_miss_lift','true_final','miss_lift')
        window=[r for r in self.rows if before<=r['pose_stamp_ns']<=self.snapshots[-1]['sim_ns']]
        physical=[native_frame(r) for r in window]
        detection=detect_miss(physical,self.params['object'],closing)
        report.update(measured_before=physical[0],measured_after_lift=physical[-1],
            detection=detection,window_frame_sha256s=[digest(r) for r in window],miss_confirmed=detection['confirmed'])
        grip('regrasp_reopen',opening)
        self.geometry.update(self.backend.snapshot())
        report['scene_after_reopen']=dict(passed=True,source='fresh native poses and verified USD FK')
        if not detection['confirmed']:self.reject('regrasp_miss_not_confirmed',detection)
        self.executed_hold=list(self.reference)
        return report

    def cycle(self):
        name=self.params['object'];self.geometry.selected=name
        self.emit(self.reference,self.executed_hold,[0.]*6,'episode_start')
        self.geometry.contact=True
        recovery=self.recover() if self.params['recovery']=='regrasp' else None
        s,pose,bbox,cups=self.current_geometry()
        opening,closing=gripper_angles(min(bbox['sx'],bbox['sy']))
        plan=make_grasp_plan(pose,bbox,current_wrist_roll=s['state'][4],
            approach_height=self.params['approach_height_m'],approach_offset=self.params['approach_offset_m'])
        yaw,solutions,_=find_reachable_grasp_yaw(plan['poses'],plan['yaw'])
        if yaw is None:self.reject('planner_reject',dict(reason='normal grasp IK unreachable'))
        self.grip(opening,'grasp_open')
        self.geometry.contact=True
        if 'approach' in solutions:self.arm(self.select(solutions['approach']),'grasp_approach')
        self.arm(self.select(solutions['final']),'grasp_descent')
        before=self.backend.snapshot()['bodies'][name]['position'][2]
        self.grip(closing,'grasp_close')
        self.geometry.bind_carried(self.backend.snapshot())
        self.arm([_grasp_home_data_target()[n] for n in JOINTS[:5]],'grasp_home')
        lifted=self.backend.snapshot()['bodies'][name]['position'][2]-before
        if lifted<.025:self.reject('insufficient_lift',dict(lift_m=lifted))
        # Same tilted over-cup IK as Quickstart; use actual block/TCP offset.
        _,_,_,cups=self.current_geometry()
        cup=cups[{'blue':'drop_1','green':'drop_2','red':'drop_3'}[self.params['color']]]
        pan=math.atan2(-cup['y'],cup['x']-X_PAN)
        half_gap=(BASELINE_JAW_GAP+JAW_GAP_RATE*self.backend.snapshot()['state'][5])/2
        solutions=geometric_ik(cup['x']+half_gap*math.sin(pan),cup['y']+half_gap*math.cos(pan),
            cup['z']+.0965/2+.05,grip_angle=math.pi/4,wrist_roll=-math.pi/2)
        for solution in solutions:solution['shoulder_pan']=pan
        self.arm(self.select(solutions),'drop_sweep')
        self.grip(opening,'drop_release')
        self.geometry.carried=None;self.geometry.contact=False
        self.arm([_grasp_home_data_target()[n] for n in JOINTS[:5]],'grasp_home_data')
        return dict(recovery_report=recovery,lift_m=lifted)
