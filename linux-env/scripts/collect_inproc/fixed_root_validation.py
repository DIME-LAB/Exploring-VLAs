"""Exercise the production ROS subscriber/controller in an owned sim window.

Run from a sourced ROS environment with --port and --output. The Kit helpers use
the already loaded production scene; they neither author nor rebuild it.
"""
import argparse
import json
from pathlib import Path
import time
import shared
import numpy as np

_probe=None


def start():
    global _probe
    import omni.usd,omni.physx,omni.timeline
    from so_arm101_dt.extension import get_instance
    from isaacsim.core.prims import RigidPrim
    from articulation_audit import readback
    from contract import JOINTS
    from safety import pose_matrix
    from pxr import PhysxSchema
    if _probe is not None:raise RuntimeError('Sweep is already active')
    extension=get_instance();art=extension._robot_view
    audit=readback(omni.usd.get_context().get_stage(),art)
    if not audit['fixed_base']:raise RuntimeError('Sweep requires a native fixed base')
    names=list(art.dof_names)
    if names!=list(JOINTS) or names!=extension._usd_joint_names:
        raise RuntimeError('Production positional joint order changed: '+repr(names))
    stage=omni.usd.get_context().get_stage()
    gpu=PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath('/physicsScene')).GetEnableGPUDynamicsAttr().Get()
    if gpu is not False:raise RuntimeError('Sweep requires CPU PhysX')
    base=RigidPrim('/World/SO_ARM101/base_link',name='fixed_root_sweep_base',reset_xform_properties=False)
    base.initialize();p,q=base.get_world_poses();anchor=pose_matrix(p[0],q[0])
    state=dict(audit=audit,joint_names=names,extension_joint_names=extension._usd_joint_names,
        initial_positions=art.get_joint_positions()[0].astype(float).tolist(),anchor=anchor.tolist(),
        samples=[],errors=[],cpu_physx=True)
    def sample(dt):
        try:
            p,q=base.get_world_poses();delta=np.linalg.inv(anchor)@pose_matrix(p[0],q[0])
            state['samples'].append(dict(dt=dt,translation_m=float(np.linalg.norm(delta[:3,3])),
                rotation_rad=float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))),
                positions=art.get_joint_positions()[0].astype(float).tolist()))
        except BaseException as exc:state['errors'].append(repr(exc))
    _probe=dict(state=state,art=art,subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(sample))
    omni.timeline.get_timeline_interface().play()
    return dict(audit=audit,joint_names=names,initial_positions=state['initial_positions'])


def read():
    import omni.graph.core as og
    art=_probe['art'];prefix='/Graph/ActionGraph_SO_ARM101/'
    return dict(positions=art.get_joint_positions()[0].astype(float).tolist(),
        targets=art.get_applied_actions().joint_positions[0].astype(float).tolist(),
        subscriber_command=np.asarray(og.Controller.attribute(prefix+'ros2_subscribe_joint_state.outputs:positionCommand').get(),float).tolist(),
        controller_indices=np.asarray(og.Controller.attribute(prefix+'articulation_controller.inputs:jointIndices').get(),int).tolist(),
        samples=len(_probe['state']['samples']))


def stop():
    global _probe
    import omni.timeline
    final=read();_probe['subscription']=None
    state=_probe['state'];_probe=None
    omni.timeline.get_timeline_interface().pause()
    state['final']=final
    for key in ('translation_m','rotation_rad'):
        state['max_'+key]=max((r[key] for r in state['samples']),default=None)
    return state


def run_ros(port,output):
    import rclpy
    from sensor_msgs.msg import JointState
    from kit_mcp.client import call
    from storage import atomic_json,confined
    output=confined(output);output.parent.mkdir(parents=True,exist_ok=True)
    prefix='import sys\nsys.path.insert(0,'+repr(str(Path(__file__).parent))+')\nimport fixed_root_validation as v\n'
    def kit(expression):return call('run_python',{'code':prefix+'result='+expression},port=port,timeout=60)['result']
    report=dict(passed=False,goals=[]);started=False;node=None
    try:
        report['initial']=kit('v.start()');started=True
        q=report['initial']['initial_positions']
        rclpy.init();node=rclpy.create_node('fixed_root_ros_sweep')
        publisher=node.create_publisher(JointState,'/joint_states',10)
        deadline=time.monotonic()+15
        while not publisher.get_subscription_count():
            if time.monotonic()>deadline:raise RuntimeError('No ROS joint-state subscriber discovered')
            rclpy.spin_once(node,timeout_sec=.05)
        for joint,offset in ((0,.12),(0,-.12),(0,0.),(1,.12),(1,-.12),(1,0.)):
            target=list(q);target[joint]+=offset
            message=JointState();message.name=['Rotation','Pitch','Elbow','Wrist_Pitch','Wrist_Roll','Jaw']
            message.position=target
            deadline=time.monotonic()+20;next_read=0.;published=0;observed=None
            while time.monotonic()<deadline:
                message.header.stamp=node.get_clock().now().to_msg();publisher.publish(message);published+=1
                rclpy.spin_once(node,timeout_sec=.03)
                if time.monotonic()>=next_read:
                    observed=kit('v.read()');next_read=time.monotonic()+.3
                    if (published>=10 and len(observed['subscriber_command'])==6 and
                        max(abs(a-b) for a,b in zip(observed['positions'],target))<.02 and
                        np.allclose(observed['subscriber_command'],target,rtol=0,atol=1e-7) and
                        np.allclose(observed['targets'],target,rtol=0,atol=1e-6)):
                        break
            else:raise RuntimeError('ROS/controller target did not converge: '+repr(observed))
            if observed['controller_indices']!=list(range(6)):raise RuntimeError('Controller positional mapping changed')
            report['goals'].append(dict(target=target,published=published,observed=observed))
        report['native']=kit('v.stop()');started=False
        native=report['native']
        if native['errors'] or not native['samples']:raise RuntimeError('Root sampler failed')
        if native['max_translation_m']>1e-7 or native['max_rotation_rad']>1e-6:
            raise RuntimeError('Fixed root moved during the pan/lift sweep')
        report['passed']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        if started:
            try:report['native']=kit('v.stop()')
            except BaseException as exc:report['stop_error']=repr(exc)
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        atomic_json(output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run_ros(args.port,args.output)
