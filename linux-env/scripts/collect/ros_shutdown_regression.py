#!/usr/bin/env python3
"""Actual shared node + Humble lifecycle, without Isaac or robot motion."""
import argparse,os,sys,threading,json
from pathlib import Path
from storage import prepare_environment,confined,atomic_json
from ros_lifecycle import retire_executor,shutdown_context


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID')!='90':raise ValueError('Requires isolated domain90')
    runtime=prepare_environment(0)
    from run_smoke import InstanceClaim,identity
    claim=InstanceClaim(runtime)
    try:
        for name in ('isaacsim-soarm101.pid','control.pid'):
            p=runtime/name
            if p.exists() and identity(int(p.read_text())):raise RuntimeError('Private instance must be down')
        output=confined(args.output);output.mkdir(exist_ok=False)
        repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'vla_SO-ARM101/src/so_arm101_control'))
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from so_arm101_control.pick_place import PickPlaceNode
        rows=[]
        for i in range(10):
            rclpy.init();node=PickPlaceNode(node_name='shutdown_regression',use_sim_time=True,joint_states_topic='/joint_states_measured');context=node.context
            actions=[type(w).__name__ for w in node.waitables if type(w).__name__=='ActionClient']
            ex=MultiThreadedExecutor(num_threads=4);ex.add_node(node);thread=threading.Thread(target=ex.spin);thread.start()
            result=retire_executor(ex,thread);node.destroy_node()
            remaining_waitables=[type(w).__name__ for w in node.waitables]
            shutdown_context(rclpy,context)
            dds=[p.read_text().strip() for p in Path('/proc/self/task').glob('*/comm') if p.read_text().strip() in ('recv','recvUC','gc','tev','dq.user','dq.builtins')]
            rows.append(dict(iteration=i,action_clients=actions,remaining_waitables=remaining_waitables,dds_threads_after=dds,**result))
            atomic_json(output/'validation.json',dict(passed=False,cycles=rows))
            assert len(actions)==2 and not remaining_waitables and not dds,rows[-1]
        atomic_json(output/'validation.json',dict(passed=True,no_isaac=True,cycles=rows))
        print(json.dumps(dict(passed=True,cycles=len(rows),output=str(output))),flush=True)
    finally:claim.close()


if __name__=='__main__':main()
