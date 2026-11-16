"""CPU fixture only: real LeRobot export/readback through the normal export entrypoint."""
import argparse,copy,importlib.util,json,os,sys,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import av
from contract import timing_audit
from storage import atomic_json,confined
from collection_status import begin,finish
from target_triplets import spec,validate_run,seal_run
from test_target_triplets import Harness


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
    certificate=begin(run);h=Harness(run,config);receipts=h.complete(recorded_ticks=100,first_offset_ticks=1)
    for index,r in enumerate(receipts):
        rows=[json.loads(line) for line in (run/f'attempt-{index:04d}/frames.jsonl').read_text().splitlines()]
        audit=timing_audit(rows,30)
        if not audit['passed']:raise ValueError(audit)
        r['timing']=audit;r['cpu_fixture_only']=True
        directory=run/f'attempt-{index:04d}';atomic_json(directory/'receipt.json',r)
        (directory/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for camera in ('top','wrist'):video(directory/(camera+'.mp4'),len(rows))
    atomic_json(run/'run.json',dict(counterfactual_triplet=config,normal_only=True,regrasp_only=False,recovery_mode='normal_only',fps=30,instance=3,cpu_fixture_only=True))
    (run/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
    seal=seal_run(run,base=out)
    atomic_json(run/'summary.json',dict(producer_invocation=certificate['invocation'],complete=True,episodes=3,end_marker='COLLECTION_FINISHED',cpu_fixture_only=True,counterfactual_triplet_seal=seal))
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
        negative={}
        protected=[run/'run.json']+[run/f'attempt-{i:04d}/{name}' for i in range(3) for name in ('receipt.json','frames.jsonl')]
        originals={p:p.read_bytes() for p in protected}
        for case in ('common_first_row','observed_rule_evidence','reused_native_chain','missing_mode','null_mode','changed_config'):
            try:
                if case=='common_first_row':
                    for i in range(3):
                        p=run/f'attempt-{i:04d}/frames.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
                        rows[0]['physical_bodies']['red_2x3_0']['position'][0]+=.25
                        p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    expected='native tick proof'
                elif case=='observed_rule_evidence':
                    p=run/'attempt-0001/receipt.json';r=json.loads(p.read_text());r['counterfactual']['observed_target_rule']['physical_sha256']='changed';atomic_json(p,r)
                    expected='Observed visible target rule evidence'
                elif case=='reused_native_chain':
                    source=json.loads((run/'attempt-0000/receipt.json').read_text())
                    for i in (1,2):
                        p=run/f'attempt-{i:04d}/receipt.json';r=json.loads(p.read_text())
                        for key in ('initial_physical','episode_start_physical','initial_restore_check'):r[key]=copy.deepcopy(source[key])
                        atomic_json(p,r)
                    expected='reused or overlapped'
                else:
                    p=run/'run.json';doc=json.loads(p.read_text())
                    if case=='missing_mode':doc.pop('counterfactual_triplet')
                    elif case=='null_mode':doc['counterfactual_triplet']=None
                    else:doc['changed_configuration']=True
                    atomic_json(p,doc);expected='configuration'
                for existing in (False,True):
                    report=out/'negative-reports'/(case+('-existing' if existing else '-fresh'))
                    args=[str(path),str(run),'--repo-id','local/cpu_counterfactual_fixture','--report-dir',str(report)]+(['--validate-existing'] if existing else [])
                    if not existing:shutil.move(str(run/'dataset'),str(run/'qualified-dataset'))
                    try:
                        with patch.object(sys,'argv',args):exporter.main()
                    except ValueError as exc:
                        assert expected in str(exc),(case,str(exc))
                        assert json.loads((report/'validation.json').read_text())['passed'] is False
                        negative[case+('-existing' if existing else '-fresh')]=str(exc)
                        if not existing:assert not (run/'dataset').exists()
                    else:raise AssertionError('Invalid triplet was exported: '+case)
                    finally:
                        if not existing:shutil.move(str(run/'qualified-dataset'),str(run/'dataset'))
            finally:
                for p,raw in originals.items():p.write_bytes(raw)
        result['negative_export_entrances']=negative
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
