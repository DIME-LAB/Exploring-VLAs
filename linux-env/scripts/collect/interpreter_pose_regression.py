"""CPU-only replay through actual restore admission in Python 3.10 and 3.12."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from storage import atomic_json,confined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=Path(__file__).resolve().parent
    out=confined(args.output)
    if out.exists():raise FileExistsError(out)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    prefix='import sys,json;sys.path.insert(0,'+repr(str(folder))+');'
    producer=prefix+'''from scene_identity import require_pose_agreement,compare
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text());s=d['snapshot'];a=d['anchor']
p=dict(against_target=require_pose_agreement(s,a),reread_snapshot=s,reread_exact_bits=compare(s,s),reread_agreement=require_pose_agreement(s,s),reread_against_target=require_pose_agreement(s,a),native_clocks_unchanged=True)
print(json.dumps(dict(version=list(sys.version_info[:2]),anchor=a,snapshot=s,check=dict(triplet_physical_state=p))))
'''
    consumer=prefix+'''from triplet_boundary import restore_admission
d=json.load(sys.stdin)
try:
 p=restore_admission(d['snapshot'],d['check'],d['anchor']);r=dict(passed=True,source_sha256=p['against_target']['exact_bits']['actual_sha256'])
except ValueError as e:r=dict(passed=False,error=str(e))
r['version']=list(sys.version_info[:2]);print(json.dumps(r))
'''
    interpreters=[sys.executable,str(args.native_python)];matrix=[];versions=set();hashes=set()
    for exe in interpreters:
        payload=json.loads(subprocess.check_output([exe,'-B','-c',producer,str(folder/'fixtures/native-pose-proof-python312.json')],env=env,text=True))
        versions.add(tuple(payload['version']))
        for reader in interpreters:
            row=dict(producer=payload['version'],consumer=reader)
            for changed in (False,True):
                data=json.loads(json.dumps(payload))
                if changed:data['check']['triplet_physical_state']['against_target']['exact_bits']['actual_sha256']='changed'
                else:
                    # Producer diagnostics are deliberately rounded differently.
                    data['check']['triplet_physical_state']['against_target']['per_body']['cup_red']['orientation_geodesic_deg']+=1e-20
                result=json.loads(subprocess.check_output([reader,'-B','-c',consumer],input=json.dumps(data),env=env,text=True))
                assert result['passed'] is (not changed),result
                if result['passed']:hashes.add(result['source_sha256'])
                row['changed_source' if changed else 'original_source']=result
            matrix.append(row)
    assert versions=={(3,10),(3,12)},versions
    assert len(hashes)==1,hashes
    atomic_json(out,dict(passed=True,real_simulation=False,versions=sorted(map(list,versions)),positive_replays=4,changed_source_refusals=4,matrix=matrix))
    print(json.dumps(dict(passed=True,positive_replays=4,changed_source_refusals=4)))


if __name__=='__main__':main()
