"""CPU fixture only: real LeRobot export/readback through the normal export entrypoint."""
import argparse,copy,importlib.util,json,os,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import av
from contract import timing_audit
from storage import atomic_json,confined
from collection_status import begin,finish
from target_triplets import spec,validate_run
from test_target_triplets import Harness
from test_contract import ContractTests
from tick_contract import STEP_NS


def video(path,count):
    container=av.open(str(path),'w');stream=container.add_stream('libx264',rate=30)
    stream.width=640;stream.height=480;stream.pix_fmt='yuv420p';stream.options={'crf':'18','g':'2','threads':'1'}
    for i in range(count):
        rgb=np.full((480,640,3),64+i%64,np.uint8);rgb[120:240,80+i:200+i]=128
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
    container.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=confined(a.out);out.mkdir(parents=True,exist_ok=False)
    config=spec(720002,'train');run=out/'datasets/counterfactual-targets/train'/config['layout_id'];run.mkdir(parents=True)
    certificate=begin(run);h=Harness(run,config);receipts=h.complete()
    for index,r in enumerate(receipts):
        initial=r['episode_start_physical'];rows=ContractTests().rows();start=initial['sim_ns']+STEP_NS
        for row in rows:
            delta=row['sim_ns']-1000000000
            for key in ('sim_ns','pose_stamp_ns','joint_stamp_ns','wrist_stamp_ns','top_stamp_ns',
                    'expert_observation_stamp_ns','application_stamp_ns','declared_application_stamp_ns'):
                row[key]=start+delta
            row.update(epoch=initial['epoch'],observation_epoch=initial['epoch'],expert_epoch=initial['epoch'],invalid_through_ns=initial['sim_ns']-STEP_NS,
                episode_id=r['episode_id'],layout_id=h.collector.triplet_session.config['layout_id'],triplet_id=h.collector.triplet_session.triplet_id,
                physical_bodies=copy.deepcopy(initial['bodies']),base_world_pose=copy.deepcopy(initial['base_world_pose']),robot_base_world_pose=copy.deepcopy(initial['robot_base_world_pose']))
            for key in ('state','physics_state','action','executed_action','expert_reference','applied_command','physics_drive_targets'):
                row[key]=[x+.125 for x in row[key]]
        rows[0]['physics_drive_targets']=list(initial['drive_targets'])
        audit=timing_audit(rows,30)
        if not audit['passed']:raise ValueError(audit)
        r['timing']=audit;r['cpu_fixture_only']=True
        directory=run/f'attempt-{index:04d}';atomic_json(directory/'receipt.json',r)
        (directory/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for camera in ('top','wrist'):video(directory/(camera+'.mp4'),len(rows))
    atomic_json(run/'run.json',dict(counterfactual_triplet=config,normal_only=True,regrasp_only=False,recovery_mode='normal_only',fps=30,instance=3,cpu_fixture_only=True))
    (run/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
    atomic_json(run/'summary.json',dict(producer_invocation=certificate['invocation'],complete=True,episodes=3,end_marker='COLLECTION_FINISHED',cpu_fixture_only=True))
    finish(run,certificate,[])
    path=Path(__file__).with_name('export.py');s=importlib.util.spec_from_file_location('triplet_checked_export',path)
    exporter=importlib.util.module_from_spec(s);s.loader.exec_module(exporter)
    result={}
    with patch.object(exporter,'validate_triplet_run',lambda run:validate_run(run,base=out)),patch.dict(os.environ,SOARM_INSTANCE_ROOT=str(out/'runtime'),CUDA_VISIBLE_DEVICES=''):
        for existing in (False,True):
            argv=[str(path),str(run),'--repo-id','local/cpu_counterfactual_fixture','--report-dir',str(out/'report')]+(['--validate-existing'] if existing else [])
            with patch.object(sys,'argv',argv):exporter.main()
            v=json.loads((out/'report/validation.json').read_text());assert v['passed'] and v['loaded'] and v['fps']==30
            assert v['episodes']==3 and v['frames']==300 and v['counterfactual_triplet']['complete']
            assert v['export_content']['video']['decoded_frame_pairs']==600
            assert 'collection_phase' not in v['features']
            result['validate_existing' if existing else 'fresh_export']=v
        receipt=run/'attempt-0001/receipt.json';original=receipt.read_text();changed=json.loads(original)
        changed['params']['task']='Wrong task';atomic_json(receipt,changed)
        try:
            with patch.object(sys,'argv',argv):exporter.main()
        except ValueError:
            stale=json.loads((out/'report/validation.json').read_text())
            assert stale['passed'] is False and stale['loaded'] is False
            result['invalid_triplet_clears_previous_pass']=True
        else:raise AssertionError('Changed triplet was accepted')
        finally:receipt.write_text(original)
        with patch.object(sys,'argv',argv):exporter.main()
    atomic_json(out/'CPU_FIXTURE_ONLY.json',dict(passed=True,real_simulation=False,fixture_training_forbidden=True,certificates=result))
    print(json.dumps(dict(passed=True,real_simulation=False,episodes=3,frames=300,decoded_frame_pairs=600)))


if __name__=='__main__':main()
