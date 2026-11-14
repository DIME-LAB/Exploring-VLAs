#!/usr/bin/env python3
"""CPU-only native MoveIt replay of final commanded geometry; never starts Isaac.

Uses an exclusive ROS domain and reconstructs each retained physical reset scene.
Historical physical observations are fixed: this checks the command gate, not yield.
"""
import argparse,copy,hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace as NS
from storage import confined,prepare_environment,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-episodes',type=int,default=50)
    args=parser.parse_args();source=confined(args.input);output=confined(args.output);output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise ValueError('Requires the assigned private domain 90')
    prepare_environment(0)
    repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'vla_SO-ARM101/src/so_arm101_control'))
    import numpy as np
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from moveit_msgs.srv import GetStateValidity
    from so_arm101_control.pick_place import PickPlaceNode
    from so_arm101_control.scene_transactions import SceneExecution
    from so_arm101_control.compute_workspace import forward_kinematics_full
    from so_arm101_control.grasp_geometry import JOINT_LIMITS
    from scene_sync import physical_pose_caches
    from scene_reconcile import SceneReconciler,ros_messages,native_scene_summary
    from query_service import StrictQueryClient
    from dart import DEFAULT_NOISE,SmoothDart,PlannerSchedule,TargetCandidate
    from contract import JOINTS,corrective_action
    from command_geometry import choose_noise_scale,final_command,CommandCollision
    rclpy.init();probe=rclpy.create_node('collection_geometry_probe')
    client=probe.create_client(GetStateValidity,'/check_state_validity')
    if client.wait_for_service(timeout_sec=3.):raise RuntimeError('Domain 90 already has MoveIt; refusing to change an existing scene')
    probe.destroy_node()
    log=(output/'move_group.log').open('w')
    process=subprocess.Popen(['nice','-n','10','ros2','launch','so_arm101_moveit_config','move_group.launch.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pids={process.pid}
    def fail(*args,**kwargs):raise RuntimeError('Offline scene/query failed: '+repr((args,kwargs)))
    node=None;executor=None;thread=None
    result=dict(scope='CPU native MoveIt final-command replay with fixed historical observations; not a physical rollout',
        source=str(source),isaac_started=False,domain=90,spacing_rad=.01,episodes=[],source_cut=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    tracked=['linux-env/scripts/collect/'+name for name in ('replay_command_geometry.py','dart.py','command_geometry.py','scene_reconcile.py','scene_sync.py','query_service.py')]
    tracked+=['vla_SO-ARM101/src/so_arm101_control/so_arm101_control/'+name for name in ('pick_place.py','grasp_geometry.py','compute_workspace.py','scene_transactions.py')]
    result['source_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in tracked}
    result['seed_scope']='phase-block seed replay; historical logs do not contain arm-operation identities'
    try:
        node=PickPlaceNode(node_name='collection_geometry_replay',execution=SceneExecution('collector',lambda epoch:nullcontext(),fail,fail))
        node._motion_generation_provider=lambda:1;node._motion_context=threading.local()
        node._physical_pose_cache=True;node._wait_arm_at_rest=lambda **kwargs:True
        node.validity_client=StrictQueryClient(node.validity_client,'offline MoveIt',fail)
        executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
        thread=threading.Thread(target=executor.spin);thread.start()
        if not node.validity_client.wait_for_service(timeout_sec=30.):raise RuntimeError('MoveIt did not start')
        reconciler=SceneReconciler(node,ros_messages(),(repo/'vla_SO-ARM101/src/so_arm101_moveit_config/config/so_arm101.srdf').read_text())
        def check(q):return node._check_state_valid_with_contacts(dict(zip(JOINTS,q)))
        limits=[JOINT_LIMITS[n] for n in JOINTS]
        receipts=[json.loads(line) for line in (source/'episodes.jsonl').read_text().splitlines()]
        for receipt in [r for r in receipts if r['accepted']][:args.max_episodes]:
            trial=receipt['params']['trial'];name=receipt['params']['object'];path=source/f'attempt-{trial:04d}'
            rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()]
            reset=receipt['episode_start_physical'];objects,cups=physical_pose_caches(reset)
            reconciler.clear();node.objects_data=objects;node._drop_data=cups
            node.objects_bbox={n:dict(zip(('sx','sy','sz'),np.ptp(corners,axis=0).tolist())) for n,corners in reset['local_corners'].items()}
            node._actual_positions=dict(zip(JOINTS,rows[0]['state']))
            node._cmd_toggle_ground_plane();node._add_cup_collision_objects(cups_dict=cups);node._add_lego_collision_objects(legos_dict=objects)
            reconciler.set_policy(['ground_plane']+list(node._cup_collision_names)+list(node._lego_collision_names))
            episode=dict(trial=trial,frames=len(rows),accepted_targets=0,refused_targets=0,preflight_refusals=[],refusals=[],noise_scales=[])
            previous=list(rows[0]['physics_drive_targets']);index=0;phase_no=0;last_block=None;removed=False
            while index<len(rows):
                end=index+1;phase=rows[index]['phase']
                while end<len(rows) and rows[end]['phase']==phase:end+=1
                block=rows[index:end]
                if phase=='grasp_move' and not removed:node._remove_single_lego_from_world(name);removed=True
                if last_block and last_block[0]['phase']=='gripper':
                    delta=last_block[-1]['expert_reference'][5]-last_block[0]['expert_reference'][5]
                    if delta<-.0005:
                        anchor=last_block[-1];native=copy.deepcopy(reset);native['bodies'][name]=anchor['target_body']
                        node.objects_data=physical_pose_caches(native)[0];node._actual_positions=dict(zip(JOINTS,anchor['state']))
                        matrix=forward_kinematics_full(anchor['state'][:5]);pose=node._mat_to_pose7(matrix)
                        node._tf_buffer=NS(lookup_transform=lambda *a,**k:NS(transform=NS(
                            translation=NS(x=pose[0],y=pose[1],z=pose[2]),rotation=NS(x=pose[3],y=pose[4],z=pose[5],w=pose[6]))))
                        if not node._attach_lego_to_gripper(name):raise RuntimeError('Offline attachment reconstruction failed')
                    elif delta>.0005 and node._attached_lego_name:node._detach_lego_sync()
                points=[NS(positions=r['expert_reference'][:5],time_from_start=NS(sec=i//30,nanosec=round((i%30)/30*1e9))) for i,r in enumerate(block)]
                duration=len(block)/30;points.append(NS(positions=block[-1]['expert_reference'][:5],time_from_start=NS(sec=int(duration),nanosec=round(duration%1*1e9))))
                schedule=PlannerSchedule(NS(points=points));noise=SmoothDart(duration,receipt['params']['noise_seed']+104729*phase_no,DEFAULT_NOISE,
                    enabled=phase not in ('gripper','grasp_home_data','home','hold'))
                node._actual_positions=dict(zip(JOINTS,block[0]['state']))
                try:scale,preflight=choose_noise_scale(schedule,noise,block[0]['expert_reference'][5],check,limits,lambda:None)
                except CommandCollision as exc:scale=0.;episode['preflight_refusals'].append(exc.evidence)
                episode['noise_scales'].append(dict(phase=phase,scale=scale))
                for i,row in enumerate(block):
                    node._actual_positions=dict(zip(JOINTS,row['state']));reference=list(row['expert_reference'])
                    offset=[scale*v for v in noise.sample(i/30)];target=[q+d for q,d in zip(reference,offset)]
                    candidate=TargetCandidate(row['epoch'],row['sim_ns'],tuple(target),(0.,)*5,tuple(offset),(0.,)*6,tuple(target),planner_reference=tuple(reference),noise_scale=scale)
                    try:
                        final_command(candidate,row['state'],previous,check,limits)
                        episode['accepted_targets']+=1
                    except CommandCollision as exc:
                        episode['refused_targets']+=1
                        if len(episode['refusals'])<8:episode['refusals'].append(dict(row=index+i,phase=phase,**exc.evidence))
                    previous=target
                last_block=block;index=end;phase_no+=1
            episode['native_scene']=native_scene_summary(reconciler.query());result['episodes'].append(episode)
            atomic_json(output/'geometry-replay.json',result);print(json.dumps({k:v for k,v in episode.items() if k not in ('native_scene','refusals','preflight_refusals')}),flush=True)
        result['complete']=True;result['frames']=sum(e['frames'] for e in result['episodes'])
    finally:
        # Track only this launch's descendants, then stop the exact owned process group.
        for item in Path('/proc').iterdir():
            if item.name.isdigit():
                try:
                    if os.getpgid(int(item.name))==process.pid:pids.add(int(item.name))
                except (ProcessLookupError,PermissionError):pass
        if executor:executor.shutdown(timeout_sec=5)
        if thread:thread.join(5)
        if node:node.running=False;node.destroy_node()
        rclpy.shutdown();os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        log.close();result['owned_pids']=sorted(pids);result['shutdown_returncode']=process.returncode
        result['surviving_owned_pids']=[p for p in pids if Path('/proc',str(p)).exists()]
        atomic_json(output/'geometry-replay.json',result)
        (output/'END').write_text('GEOMETRY_REPLAY_FINISHED\n' if result.get('complete') else 'GEOMETRY_REPLAY_INCOMPLETE\n')
    if result['surviving_owned_pids']:raise RuntimeError('Offline MoveIt children survived shutdown')


if __name__=='__main__':main()
