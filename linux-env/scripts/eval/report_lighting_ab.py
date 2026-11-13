"""Paired inverse-lighting comparison against the fixed 1.9 RTC-off baseline."""
from pathlib import Path
import argparse,json,csv,math,statistics
from report_results import wilson
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
root=parser.parse_args().root;groups={};configs={}
for label in ['baseline','treatment']:
 p=root/'raw'/label;configs[label]=json.loads((p/'run_config.json').read_text())
 rs=[json.loads(line) for line in (p/'results.jsonl').read_text().splitlines()]
 assert len(rs)==10 and sorted(r['seed'] for r in rs)==list(range(900000,900010))
 for r in rs:
  assert r['outcome'] in ['success','grasped-not-placed','no-grasp']
  assert r['client']['missed_action_ticks_after_startup']==0
  costs=sorted(r['client']['send_action_costs_ms']);assert costs[math.ceil(.95*len(costs))-1]<5
  assert abs(r['sim_cap_hz']-17.4)<1e-4
  r['characterization']=json.loads((p/r['episode']/'characterization.json').read_text())
 c=configs[label];assert c['fps']==17.4 and c['sim_rate_hz']==17.4 and not c['rtc_enabled']
 assert c['joint_states_topic']=='/joint_states' and c['actions_per_chunk']==50 and c['chunk_size_threshold']==.5
 groups[label]={r['seed']:r for r in rs}
assert configs['baseline']['pretrained']==configs['treatment']['pretrained']
assert configs['baseline']['fork_commit']==configs['treatment']['fork_commit']
paired=[]
for seed in range(900000,900010):
 a=groups['baseline'][seed];b=groups['treatment'][seed];assert a['randomize']['bodies']==b['randomize']['bodies']
 assert b['observation_transform']['enabled'] and b['observation_transform']['divisor']==[1.12162,1.11828,1.11650]
 d=dict(seed=seed)
 for label,r in [('baseline',a),('treatment',b)]:
  ch=r['characterization'];d.update({label+'_outcome':r['outcome'],label+'_closest_tcp_blue_mm':1000*ch['closest_blue_center_distance_m'],label+'_closing_motion_ge2deg':ch['closing_motion_ge2deg'],label+'_minimum_gripper_deg':ch['gripper_min_deg'],label+'_any_sustained_object_lift':ch['any_sustained_lift_24mm'],label+'_effective_hz':r['effective_action_fps'],label+'_empty_ticks_pct':r['empty_ticks_pct'],label+'_load_start':r['host_load_start'][0],label+'_load_end':r['host_load_end'][0]})
 d['delta_closest_mm']=d['treatment_closest_tcp_blue_mm']-d['baseline_closest_tcp_blue_mm'];d['delta_effective_hz']=b['effective_action_fps']-a['effective_action_fps'];d['delta_empty_pct']=b['empty_ticks_pct']-a['empty_ticks_pct'];paired.append(d)
with (root/'paired-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
summary={}
for label,rs in groups.items():
 values=list(rs.values());k=sum(r['outcome']=='success' for r in values);g=sum(r['outcome']=='grasped-not-placed' for r in values)
 loads=[r[key][0] for r in values for key in ['host_load_start','host_load_end']]
 summary[label]=dict(success=k,grasp_only=g,n=10,wilson95=wilson(k,10),closing_motion=sum(r['characterization']['closing_motion_ge2deg'] for r in values),any_lift=sum(r['characterization']['any_sustained_lift_24mm'] for r in values),mean_closest_mm=statistics.mean(r['characterization']['closest_blue_center_distance_m']*1000 for r in values),mean_effective_hz=statistics.mean(r['effective_action_fps'] for r in values),mean_empty_pct=statistics.mean(r['empty_ticks_pct'] for r in values),load_range=[min(loads),max(loads)])
summary['paired']=dict(mean_delta_closest_mm=statistics.mean(d['delta_closest_mm'] for d in paired),median_delta_closest_mm=statistics.median(d['delta_closest_mm'] for d in paired),closer_ge1mm=sum(d['delta_closest_mm']<=-1 for d in paired),farther_ge1mm=sum(d['delta_closest_mm']>=1 for d in paired),success_difference=(summary['treatment']['success']-summary['baseline']['success'])/10)
summary['actual_child_transform_proof']=groups['treatment'][900000]['client']['observation_transform_proof']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
s=summary;pa=s['paired'];a=s['baseline'];b=s['treatment']
first=(f'The inverse-gain treatment produced {b["success"]}/10 placements and {b["grasp_only"]}/10 blue-lift-only episodes, versus {a["success"]}/10 and {a["grasp_only"]}/10 in the 1.9 RTC-off baseline. '
       f'Mean paired closest commanded-TCP-to-blue-centre distance changed by {pa["mean_delta_closest_mm"]:+.2f} mm (negative is closer); {pa["closer_ge1mm"]} seeds became at least 1 mm closer and {pa["farther_ge1mm"]} at least 1 mm farther. '
       'This small historical-baseline comparison does not establish a causal lighting effect or equivalence.')
lines=['# 1.10a lighting A/B','',first,'','| Condition | Placement | 95% Wilson interval | Blue lift only | Closing motion | Any object lift | Mean closest FK distance mm | Effective Hz | Empty ticks % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for label in ['baseline','treatment']:
 x=s[label];lo,hi=x['wilson95'];lines.append(f'| {label} | {x["success"]}/10 | {lo:.1%}–{hi:.1%} | {x["grasp_only"]}/10 | {x["closing_motion"]}/10 | {x["any_lift"]}/10 | {x["mean_closest_mm"]:.2f} | {x["mean_effective_hz"]:.3f} | {x["mean_empty_pct"]:.2f}% |')
lines+=['','## Paired scenes','','| Seed | Baseline closest mm | Corrected closest mm | Δ mm | Baseline / corrected closing motion | Baseline / corrected lift |','|---|---:|---:|---:|---|---|']
for d in paired:lines.append(f'| {d["seed"]} | {d["baseline_closest_tcp_blue_mm"]:.2f} | {d["treatment_closest_tcp_blue_mm"]:.2f} | {d["delta_closest_mm"]:+.2f} | {d["baseline_closing_motion_ge2deg"]} / {d["treatment_closing_motion_ge2deg"]} | {d["baseline_any_sustained_object_lift"]} / {d["treatment_any_sustained_object_lift"]} |')
lines+=['','## Protocol and limits','','NEW checkpoint `/home/aaugus11/w_ckpt/smolvla-base-ft-rec222812-20k-s42-20261113`, task “Pick a blue lego and place it in blue cup”, state `/joint_states`, client/sim cap17.4 Hz, RTC off, isolated capture/FIFO10,50 actions per chunk and refill threshold0.5. Both conditions use seeds900000–900009 and exactly matching saved physical block/cup poses. Each episode homes first and runs up to45 wall seconds. All treatment episodes use `--dataset-brightness`: round/clip live RGB divided by[1.12162,1.11828,1.11650], applied to both policy cameras before the unchanged RPC. The physical scene, viewport images and state/task are not photometrically modified. The flag defaults off and is recorded in every result. No fork file changed.','','The saved-frame implementation check and the first actual isolated-child proof both map top-table median[83,104,115] to[74,93,103]. Both camera hashes change; raw arrays are unchanged. See `provenance/transform-verification.json`, `calibration-check.png` and ep00’s `observation_transform_proof`. The wrist uses the same gains as an explicit sensitivity assumption, not an independent wrist calibration.','','Every accepted episode has zero missed scheduling slots, no stale-camera abort, correct cap read-back and actuator-send p95<5ms. Empty queues remain measured outcomes; an effective-Hz band is not an exclusion rule. CPU load is disclosed below.','','Closest distances use the production URDF fixed-jaw-tip FK on recorded mock-controller-reference joints; they are command proxies, not historical measured TCP. With two blue legos, closest distance is the minimum over either blue centre; selected IDs are retained in the characterization sidecars. Closing motion means a decrease>=2deg from a preceding opening; aperture/closure positions and any sustained24mm/0.5s lift of any colour are in `failure-characterization.csv` and sidecars. The 1.9 static FK validation does not bound dynamic tracking lag.','','The baseline was recorded earlier, not interleaved/randomized with treatment. Policy flow noise was not paired across these historical runs. Timing, background load and the cost of correction can also differ. With ten scenes, zero successes still has a27.8% Wilson upper bound; mixed distance changes or zero observed success differences do not prove lighting irrelevant. No other photometric family, gain strength or deployment knob was searched.','','| Condition | Boundary1-minute CPU load range (28 CPUs) |','|---|---:|']
for label in ['baseline','treatment']:lo,hi=s[label]['load_range'];lines.append(f'| {label} | {lo:.2f}–{hi:.2f} |')
lines+=['','## Evidence','','- `paired-results.csv`: all per-seed outcomes, distance/closure/lift and Hz/starvation/load measurements.','- `raw/treatment/`: all10 full videos and pose/chunk/frame/fault/source/config receipts.','- `raw/baseline/`: frozen1.9 reference metadata and poses; original full videos remain under phase1/1.9-eval/raw/new-17.4-rtc-off/.','- `clips/`: short baseline/corrected videos for seeds900000,900003 and900006; viewport pixels remain raw live appearance.','- `summary.json`, transform proof and analysis scripts: reproducibility evidence.','']
(root/'README.md').write_text('\n'.join(lines))
print(json.dumps(summary))
