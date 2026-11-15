"""Audit and summarize paired five-noise-seed CPU language interventions."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics
import numpy as np
from direction_metrics import angle_degrees,motion_directions,seed_consistency
from behaviour import fk,rotate
from checkpoint_identity import checkpoint_label

COLORS=('red','green','blue');SEEDS=list(range(42,47))
ANCHORS={('seen-training',s) for s in range(530000,530003)}|{('held-out-pilot',s) for s in range(350000,350003)}
WEIGHTS={20000:'776744f3b09f3fd420a5812f0c636bcc31b5526ae3373b5ac244ae0b990dd173',30000:'87bc05bd7b2a8cab75febca69754dc23c16c116e7bf806bc5d6212b09847d31c'}
def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--reference-20k',type=Path,required=True);p.add_argument('--models',nargs='+',default=['020000','030000']);p.add_argument('--expected-models',type=Path);a=p.parse_args();root=a.root;model_keys=a.models
    expected={f'{step:06d}':dict(step=step,sha256=digest,path=f'/mnt/a4500-backup/exploring-vlas/eval/checkpoints/smolvla-phase3-normal-60k-s42-20261114/checkpoints/{step:06d}/pretrained_model') for step,digest in WEIGHTS.items()}
    if a.expected_models:
        for key,value in json.loads(a.expected_models.read_text()).items():
            if key in expected:assert value==expected[key],'Pinned primary checkpoint identity differs'
            else:expected[key]=value
    assert len(model_keys)==len(set(model_keys)) and model_keys and all(k in expected for k in model_keys)
    assert not (root/'provenance/delivery-manifest.json').exists(),'Finalized delivery'
    labels={k:checkpoint_label(expected[k]['path']) for k in model_keys}
    cases={};flat=[];blocks=[];consistency=[];summaries={};controls=[];native_controls=[];pixels={};tokens={};poses={};case_hashes=set()
    for key in model_keys:
        step=expected[key]['step']
        folder=root/key;doc=json.loads((folder/'colour-swap.json').read_text());assert doc['checkpoint_step']==step and doc['flow_noise_seeds']==SEEDS
        assert doc['weights_sha256']==expected[key]['sha256'] and doc['checkpoint_path']==expected[key]['path'] and doc['checkpoint_label']==labels[key];case_hashes.add(doc['case_manifest_sha256'])
        found={(r['kind'],r['seed'],r['sampling_seed']):r for r in doc['results']};assert len(found)==len(doc['results'])==30,'Expected exactly 30 unique anchor/noise records'
        repeats=[r['identical_prompt_repeat_max_abs_deg'] for r in doc['results'] if r['identical_prompt_repeat_max_abs_deg'] is not None];assert len(repeats)==1 and repeats[0]<1e-6,'Missing/failed identical red repeat'
        assert set(found)=={(*anchor,noise) for anchor in ANCHORS for noise in SEEDS}
        cases[key]=found;allpairs=[]
        for case_key,r in found.items():
            kind,seed,noise=case_key;assert r['anchor_frame']==0 and r['pixels_and_state_fixed']
            anchor=(kind,seed);fingerprint=(r['input_sha256'],r['state_degrees'])
            if anchor in pixels:assert pixels[anchor]==fingerprint
            else:pixels[anchor]=fingerprint
            pose=(r['home_tcp_world_m'],r['block_positions_world_m'])
            if anchor in poses:assert poses[anchor]==pose
            else:poses[anchor]=pose
            file=folder/f'colour-swap-{kind}-{seed}-noise-{noise}.npz';arr=np.load(file)
            assert np.array_equal(arr['state'],r['state_degrees'])
            if key=='020000' and noise==42:
                ref=np.load(a.reference_20k/f'colour-swap-{kind}-{seed}.npz')
                error=max(float(np.max(np.abs(arr['pred_'+c]-ref['pred_'+c]))) for c in COLORS)
                assert error<1e-6,(seed,error);controls.append(dict(seed=seed,reference_20k_seed42_max_abs_deg=error))
            if r['identical_prompt_repeat_max_abs_deg'] is not None:assert r['identical_prompt_repeat_max_abs_deg']<1e-6
            if noise==42:
                assert r['source_prompt_control_max_abs_deg']<1e-6
                native_controls.append(dict(checkpoint_key=key,checkpoint_step=step,seed=seed,max_abs_deg=r['source_prompt_control_max_abs_deg']))
            assert r['native_camera_order']==['observation.images.top','observation.images.wrist']
            assert all(r['prompts'][c]['normalized_state']==r['prompts']['red']['normalized_state'] for c in COLORS)
            for color in COLORS:
                prompt=r['prompts'][color];assert prompt['task']==f'Pick a {color} lego and place it in {color} cup'
                tokenkey=(kind,seed,color);token=(prompt['tokens'],prompt['token_mask'])
                if tokenkey in tokens:assert tokens[tokenkey]==token
                else:tokens[tokenkey]=token
                pred,tcp=arr['pred_'+color],arr['tcp_'+color];assert pred.shape==(50,6) and tcp.shape==(50,3) and np.isfinite(pred).all() and np.isfinite(tcp).all()
                base=r['robot_base_world_pose'];local=fk.forward_kinematics_batch(np.radians(pred[:,:5]));rebuilt=np.array([np.array(rotate(base['orientation'],pt.tolist()))+base['position'] for pt in local]);assert np.allclose(rebuilt,tcp,rtol=0,atol=1e-10)
                geometry=motion_directions(r['home_tcp_world_m'],tcp,r['block_positions_world_m'],color)
                assert geometry==prompt['geometry']
                base=dict(checkpoint_key=key,checkpoint_step=step,kind=kind,seed=seed,flow_noise_seed=noise,prompt_color=color)
                base.update(expert_target_body=r['expert_target_body'])
                flat.append(dict(base,endpoint_world_m=tcp[-1].tolist(),motion_mm=geometry['motion_mm'],horizontal_motion_mm=geometry['horizontal_motion_mm'],path_length_mm=geometry['path_length_mm'],xy_target_angle_deg=geometry['xy']['target_angle_deg'],xy_other_angle_deg=geometry['xy']['other_angle_deg'],xy_target_margin_deg=geometry['xy']['margin_deg'],xy_requested_winner=geometry['xy']['requested_winner'],xy_winner_blocks=geometry['xy']['winner_blocks'],xyz_requested_winner=geometry['xyz']['requested_winner'],low_horizontal_motion=geometry['low_horizontal_motion'],requested_already_aligned_xy=geometry['requested_already_aligned_xy']))
                for b in geometry['blocks']:blocks.append(dict(base,**b))
            for pair in r['pairs']:
                l,h=pair['prompts'];changed=[i for i,(x,y) in enumerate(zip(r['prompts'][l]['tokens'],r['prompts'][h]['tokens'])) if x!=y]
                assert changed==pair['changed_token_positions']==[2,9]
                assert r['prompts'][l]['token_mask']==r['prompts'][h]['token_mask']
                assert np.isclose(pair['arm_l1_deg'],np.abs(arr['pred_'+l]-arr['pred_'+h])[:,:5].mean())
                assert np.isclose(pair['tcp_endpoint_shift_mm'],np.linalg.norm(arr['tcp_'+l][-1]-arr['tcp_'+h][-1])*1000)
                allpairs.append(pair)
        for kind,seed in sorted(ANCHORS):
            for color in COLORS:
                motions=[found[(kind,seed,n)]['prompts'][color]['geometry'] for n in SEEDS]
                consistency.append(dict(checkpoint_key=key,checkpoint_step=step,kind=kind,seed=seed,prompt_color=color,**seed_consistency(motions)))
        fr=[r for r in flat if r['checkpoint_key']==key];cs=[r for r in consistency if r['checkpoint_key']==key]
        summaries[key]=dict(weights_sha256=doc['weights_sha256'],n_chunks=len(fr),requested_xy_wins=sum(r['xy_requested_winner'] is True for r in fr),classified_xy_n=sum(r['xy_requested_winner'] is not None for r in fr),low_horizontal_n=sum(r['low_horizontal_motion'] for r in fr),mean_prompt_pair_arm_l1_deg=statistics.mean(r['arm_l1_deg'] for r in allpairs),mean_prompt_pair_endpoint_mm=statistics.mean(r['tcp_endpoint_shift_mm'] for r in allpairs),mean_seed_endpoint_spread_rms_mm=statistics.mean(r['endpoint_spread_rms_mm'] for r in cs),mean_direction_resultant=statistics.mean([r['direction_resultant'] for r in cs if r['direction_resultant'] is not None]) if any(r['direction_resultant'] is not None for r in cs) else None,pair_endpoint_means_mm={'/'.join(combo):statistics.mean(r['tcp_endpoint_shift_mm'] for r in allpairs if r['prompts']==list(combo)) for combo in [('red','green'),('red','blue'),('green','blue')]})
        summaries[key].update(wrong_colour_directions=sum(c['wrong_colour_n'] for c in cs),ambiguous_colour_directions=sum(c['ambiguous_colour_n'] for c in cs),requested_colour_multimodal_groups=sum(len(c['requested_winner_distinct_blocks'])>1 for c in cs))
        summaries[key].update(undefined_requested_bearing_n=sum(c['undefined_requested_bearing_n'] for c in cs),small_margin_under5deg_n=sum(c['small_margin_under5deg_n'] for c in cs))
        summaries[key].update(checkpoint_path=doc['checkpoint_path'],checkpoint_label=labels[key],checkpoint_step=step)
        s=summaries[key];assert sum(s[k] for k in ('requested_xy_wins','wrong_colour_directions','ambiguous_colour_directions','low_horizontal_n','undefined_requested_bearing_n'))==90
    assert len(flat)==90*len(model_keys) and len(controls)==(6 if '020000' in model_keys else 0) and len(native_controls)==6*len(model_keys) and len(case_hashes)==1
    layout_gaps=[]
    for (kind,seed),(home,positions) in sorted(poses.items()):
        entries=[];home=np.array(home)
        for left,p1 in positions.items():
            for right,p2 in positions.items():
                if right<=left or left.split('_')[0]==right.split('_')[0]:continue
                angle=angle_degrees((np.array(p1)-home)[:2],(np.array(p2)-home)[:2])
                if angle is not None:entries.append((angle,left,right))
        angle,left,right=min(entries);layout_gaps.append(dict(kind=kind,seed=seed,min_crosscolour_bearing_gap_deg=angle,blocks=[left,right]))
    write_csv(root/'per-prediction.csv',flat);write_csv(root/'per-block-angles.csv',blocks);write_csv(root/'noise-consistency.csv',consistency)
    (root/'summary.json').write_text(json.dumps(dict(checkpoints=summaries,controls=controls,native_source_prompt_controls=native_controls,layout_bearing_gaps=layout_gaps,case_manifest_sha256=next(iter(case_hashes)),input_hashes_identical=True,tokens_differ_only_positions=[2,9],flow_noise_seeds=SEEDS,minimum_directional_motion_mm=1),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,len(model_keys),figsize=(5.5*len(model_keys),23),squeeze=False);paints=dict(red='#bc4141',green='#299364',blue='#397dc1')
    for i,(kind,seed) in enumerate(sorted(ANCHORS)):
        for j,key in enumerate(model_keys):
            ax=axes[i,j];r=cases[key][(kind,seed,42)];home=np.array(r['home_tcp_world_m']);ax.scatter(0,0,c='black',marker='x',s=50,label='HOME TCP')
            for name,pos in r['block_positions_world_m'].items():
                offset=(np.array(pos)-home)*1000;color=name.split('_')[0];ax.scatter(*offset[:2],color=paints[color],marker='s',s=30);ax.annotate(name,offset[:2],fontsize=6)
            for noise in SEEDS:
                for color in COLORS:
                    motion=np.array(cases[key][(kind,seed,noise)]['prompts'][color]['displacement_world_m'])*1000
                    ax.annotate('',xy=motion[:2],xytext=(0,0),arrowprops=dict(arrowstyle='->',color=paints[color],alpha=.5,lw=1.2))
            ax.set(title=f'{labels[key]} · {kind} · {seed}',xlabel='World ΔX (mm)',ylabel='World ΔY (mm)');ax.axis('equal');ax.grid(alpha=.2)
    fig.suptitle('Five flow-noise seeds per colour prompt · net 50-action TCP displacement\nSquares: both blocks of each colour; arrows: prompt colour; same HOME pixels/state across all predictions.');fig.tight_layout(rect=(0,0,1,.97));fig.savefig(root/'directions.png',dpi=150);plt.close(fig)
    table=['| Checkpoint | Requested colour wins XY direction | Low XY motion | Pairwise prompt arm L1 | Pairwise endpoint separation | Noise endpoint RMS spread | Mean direction resultant R |','|---|---|---|---|---|---|---|']
    for key in model_keys:
        step=expected[key]['step']
        s=summaries[key];r_display='—' if s['mean_direction_resultant'] is None else f"{s['mean_direction_resultant']:.3f}";table.append(f"| {labels[key]} | {s['requested_xy_wins']}/{s['classified_xy_n']} classified | {s['low_horizontal_n']}/90 | {s['mean_prompt_pair_arm_l1_deg']:.3f}° | {s['mean_prompt_pair_endpoint_mm']:.2f} mm | {s['mean_seed_endpoint_spread_rms_mm']:.2f} mm | {r_display} |")
    detail=['| Checkpoint | Anchor | Prompt | Target wins / classified seeds | R | Maximum between-seed angle | Endpoint RMS spread |','|---|---|---|---|---|---|---|']
    for c in consistency:
        display=lambda k:'—' if c[k] is None else f'{c[k]:.2f}'
        detail.append(f"| {labels[c['checkpoint_key']]} | {c['seed']} ({c['kind']}) | {c['prompt_color']} | {c['requested_direction_wins']}/{c['requested_win_denominator']} | {display('direction_resultant')} | {display('max_seed_direction_angle_deg')}° | {c['endpoint_spread_rms_mm']:.2f} mm |")
    headline='; '.join(f"{labels[k]}: {summaries[k]['requested_xy_wins']}/{summaries[k]['classified_xy_n']} classified XY directions" for k in model_keys)
    text=f'''CPU-only HOME colour-direction probes: {headline}. These are {90*len(model_keys)} unscored predictions on six fixed observations with three prompts and five paired flow-noise seeds, not independent scenes or closed-loop successes.

{chr(10).join(table)}

![Directions across five noise seeds](directions.png)

Seen anchors are training batch rtx6000-normal-i3-b00 seeds 530000–530002; held-out anchors are pilot-3.1-v4 seeds 350000–350002, absent from the 900 training layouts. All use frame zero and native camera/state/model processors, eager CPU, RTC disabled. No training-box GPU was used. Identical camera-tensor/state hashes, home TCP, block positions, tokens and masks are checked across noise seeds/checkpoints. Colour tokens differ at positions 2 and 9 only. Flow seeds 42–46 are reset before every prediction and paired across prompts/checkpoints. The first identical red prediction repeat is exact at each checkpoint, and native source-prompt controls are repeated at all six anchors. {'All six 20k seed42 prompt triplets reproduce the prior finalized HOME probe within 1e-6 degrees.' if '020000' in model_keys else 'This batch contains no primary20k predictions; no20k reference repeat is claimed.'} Controls are in summary.json.

For each predicted 50-action chunk, production FK transforms the endpoint and path to the recorded robot base's world frame. The angle is between net endpoint displacement from HOME and the HOME-to-block-centre bearing. [per-block-angles.csv](per-block-angles.csv) gives 3D and XY angles to **every** block, endpoint distance and distance reduction; the requested target may be either of two same-colour blocks. XY classification compares the smallest angle to a requested block against every other block, with a tie across colours not a unique requested win. Motions under 1 mm in XY are unclassified horizontally, even if they descend vertically; full 3D angles remain available when 3D motion exceeds 1 mm. Already-aligned XY blocks are marked and have no defined XY bearing. Angles and net direction are diagnostics, not contact, grasp success or a trajectory-optimality criterion.

[noise-consistency.csv](noise-consistency.csv) gives each anchor/prompt's five sample choices, resultant R (norm of mean unit XY directions), maximum pairwise seed angle and endpoint RMS spread. R near one means similar headings; R near zero can reflect alternatives, including two valid blocks of the **same** requested colour. Do not infer wrong-colour averaging from low R alone: inspect per-seed winning block sets. Net displacement can hide curved paths; full TCP paths and action arrays are saved in each checkpoint's NPZs and path lengths appear in [per-prediction.csv](per-prediction.csv). Prompt-pair L1/separation and per-pair means are in summary.json. Different scene layouts confound comparisons against different experts; no same-layout counterfactual expert trajectory is claimed.

{chr(10).join(detail)}

The scored MAIN runs remain separate and use their frozen RTC-on/off deployment. This intervention checks language/noise sensitivity and does not isolate a frozen-VLM, multimodality, training or deployment cause. Reproduce with delivered open_loop.py (`--home --colour-swap --flow-seeds 42 43 44 45 46 --out <new-directory>`) and report_directions.py; saved code and model/input hashes are delivered alongside results.
'''
    partitions=['| Checkpoint | Requested win | Wrong colour | Ambiguous colour tie | Low XY motion | Requested bearing undefined | Absolute margin <5° |','|---|---|---|---|---|---|---|']
    for key in model_keys:
        step=expected[key]['step']
        s=summaries[key];partitions.append(f"| {labels[key]} | {s['requested_xy_wins']} | {s['wrong_colour_directions']} | {s['ambiguous_colour_directions']} | {s['low_horizontal_n']} | {s['undefined_requested_bearing_n']} | {s['small_margin_under5deg_n']} |")
    text+='\n'+'\n'.join(partitions)+'\n\nThe first five outcome categories partition all90 predictions at each checkpoint; small margin overlaps them. The 5° margin flag is descriptive and does not alter the exact angular winner. Cross-colour bearing gaps and all native source-prompt controls are in summary.json. Small gaps make nearest-bearing wins fragile: '+ '; '.join(f"anchor{g['seed']}: {g['min_crosscolour_bearing_gap_deg']:.2f}° ({'/'.join(g['blocks'])})" for g in layout_gaps)+'.\n'
    (root/'README.md').write_text(text);print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
