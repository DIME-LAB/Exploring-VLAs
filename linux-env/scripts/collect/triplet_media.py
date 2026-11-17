"""Production first-observation/label gate on original recorder videos.

Pure measurements do not certify provenance. The CLI first validates the sealed
native group and refuses terminal markers; the runner binds its invocation and
the returned file hashes before selecting a group.
"""
import argparse,hashlib,itertools,json,math,re,struct
from pathlib import Path
import numpy as np

POLICY=dict(version=1,min_each_camera_psnr_db=40.,prefix_frames=50,
    min_departure_action_difference_rad=1e-6,exact_first_physical_hash=True,exact_first_state=True,
    video_source='original recorder CRF18, no shared/replaced pixels')


def measure(hashes,states,actions,images):
    states=np.asarray(states,dtype=np.float64);actions=np.asarray(actions,dtype=np.float64)
    if len(hashes)!=3 or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in hashes):raise ValueError('Three native physical hashes required')
    if states.shape!=(3,6) or actions.shape!=(3,50,6) or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Incomplete/nonfinite first state or first50 clean actions')
    if set(images)!={'top','wrist'}:raise ValueError('Both cameras required')
    reasons=[];pairs=[];labels=[]
    if len(set(hashes))!=1:reasons.append('exact_identity')
    bits=[b''.join(struct.pack('>d',float(v)) for v in row) for row in states]
    if len(set(bits))!=1 and 'exact_identity' not in reasons:reasons.append('exact_identity')
    for camera,values in images.items():
        if len(values)!=3 or any(v.shape!=(480,640,3) or v.dtype!=np.uint8 for v in values):raise ValueError('Invalid RGB camera layout')
        for a,b in itertools.combinations(range(3),2):
            delta=values[a].astype(np.float64)-values[b];mse=float(np.mean(delta*delta))
            psnr=10*math.log10(255**2/mse) if mse else None
            passed=psnr is None or psnr>=POLICY['min_each_camera_psnr_db']
            pairs.append(dict(camera=camera,members=[a,b],psnr_db=psnr,identical_rgb=mse==0.,mae=float(np.mean(np.abs(delta))),passed=passed))
            if not passed and 'image_psnr' not in reasons:reasons.append('image_psnr')
    for a,b in itertools.combinations(range(3),2):
        delta=np.max(np.abs(actions[a]-actions[b]),axis=1);different=np.flatnonzero(delta>POLICY['min_departure_action_difference_rad'])
        labels.append(dict(members=[a,b],frame0_max_abs_deg=math.degrees(float(delta[0])),
            first_different_frame=int(different[0]) if len(different) else None,max_abs_first50_deg=math.degrees(float(np.max(delta)))))
        if not len(different) and 'departure_labels' not in reasons:reasons.append('departure_labels')
    return dict(passed=not reasons,reasons=reasons,policy=POLICY,physical_hashes=hashes,
        first_state_bits_equal=len(set(bits))==1,image_pairs=pairs,label_pairs=labels)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def inspect_run(run):
    """Recompute production admission from the sealed raw group, not its flags."""
    from storage import confined
    from target_triplets import validate_run
    from collection_status import reject_terminal
    import av
    run=confined(run)
    from triplet_artifacts import checked_tree
    checked_tree(run)
    reject_terminal(run);group=validate_run(run);images=dict(top=[],wrist=[]);states=[];actions=[];files={}
    for i in range(3):
        attempt=run/f'attempt-{i:04d}'
        with (attempt/'frames.jsonl').open() as stream:rows=[json.loads(next(stream)) for _ in range(50)]
        states.append(rows[0]['state']);actions.append([r['action'] for r in rows])
        for name in ('receipt.json','frames.jsonl','top.mp4','wrist.mp4','first-recorded-native.json'):
            path=attempt/name;files[str(path.relative_to(run))]=sha(path)
        for camera in images:
            with av.open(str(attempt/(camera+'.mp4'))) as video:images[camera].append(next(video.decode(video=0)).to_ndarray(format='rgb24'))
    result=measure(group['first_recorded_physical_sha256'],states,actions,images)
    result.update(layout_seed=group['layout_seed'],split=group['split'],triplet_id=group['triplet_id'],
        validated_group_sha256=sha(run/'counterfactual-triplet.json'),raw_files_sha256=files)
    reject_terminal(run)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from storage import confined,atomic_json
    output=confined(args.output)
    if output.exists():raise FileExistsError('Fresh production-media receipt required')
    result=inspect_run(args.run);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(passed=result['passed'],reasons=result['reasons'])))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
