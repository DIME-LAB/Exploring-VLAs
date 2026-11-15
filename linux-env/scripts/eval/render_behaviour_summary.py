"""Failure characterization for any scored arm, including missing closures."""
from pathlib import Path
import argparse,json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from checkpoint_identity import checkpoint_label

p=argparse.ArgumentParser();p.add_argument('root',type=Path);root=p.parse_args().root
assert not (root/'provenance/delivery-manifest.json').exists(),'Finalized delivery'
folder=root/'behaviour';rows=json.loads((folder/'episodes.json').read_text());experts=json.loads((folder/'training-experts.json').read_text());summary=json.loads((root/'summary.json').read_text());rtc=summary['contract']['rtc_enabled'];label=f"{checkpoint_label(summary['contract']['pretrained'])} RTC {'on' if rtc else 'off'}"
def mean(rows,key,scale=1):
    values=[r[key]*scale for r in rows if r.get(key) is not None]
    return 'undefined' if not values else f'{statistics.mean(values):.2f} (N={len(values)})'
closed=[r for r in rows if r['ever_closed_ge2deg']];near=sum(r.get('target_nearer_at_closing') is True for r in closed)
fig,axes=plt.subplots(1,4,figsize=(15,4.5))
for ax,(key,scale,title) in zip(axes,[('closing_distance_m',1000,'First closure distance (mm)'),('min_tcp_target_m',1000,'Minimum target distance (mm)'),('tcp_dwell_pct',1,'Low-motion dwell (%)'),('closing_time_sim_s',1,'First closure time (sim s)')]):
    for i,(rs,color) in enumerate([(experts,'#3d9465'),(rows,'#bd5748')]):
        values=[r[key]*scale for r in rs if r.get(key) is not None]
        if values:
            ax.scatter(i+np.linspace(-.1,.1,len(values)),values,color=color,alpha=.65,s=20);ax.plot([i-.17,i+.17],[statistics.mean(values)]*2,color='#222')
    ax.set_xticks([0,1],['Expert (5)',label]);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same measured-state FK and diagnostics · layouts differ between expert and policy\nDots include invalid attempts; missing closures are omitted only from closure metrics.');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(folder/'expert-comparison.png',dpi=170);plt.close(fig)
table=['| Seed | Colour | Minimum target / other (mm) | Closed ≥2° | Close time (sim s) | Close XYZ world (mm) | Close target distance (mm) | Nearest colour at close | Low-motion dwell (%) |','|---|---|---|---|---|---|---|---|---|']
for r in rows:
    xyz='/'.join(f'{1000*v:.1f}' for v in r['closing_tcp_world']) if r.get('closing_tcp_world') else '—'
    fmt=lambda key,scale=1:'—' if r.get(key) is None else f'{scale*r[key]:.2f}'
    table.append(f"| {r['seed']} | {r['color']} | {fmt('min_tcp_target_m',1000)} / {fmt('min_tcp_other_m',1000)} | {r['ever_closed_ge2deg']} | {fmt('closing_time_sim_s')} | {xyz} | {fmt('closing_distance_m',1000)} | {r.get('nearest_colour_at_closing','—')} | {fmt('tcp_dwell_pct')} |")
text=f'''{label} closes by at least 2° in {len(closed)}/30 episodes; the requested colour is nearest at the first closing movement in {near}/{len(closed)} closures. Mean target distance at closure is {mean(rows,'closing_distance_m',1000)} mm versus {mean(experts,'closing_distance_m',1000)} mm for five training experts. Mean low-motion TCP dwell is {mean(rows,'tcp_dwell_pct')}% versus expert {mean(experts,'tcp_dwell_pct')}%. These are geometric failure diagnostics, not grasp/contact labels.

![Policy and training expert diagnostics](expert-comparison.png)

Minimum target/other-block distances average {mean(rows,'min_tcp_target_m',1000)} / {mean(rows,'min_tcp_other_m',1000)} mm. First closure averages {mean(rows,'closing_time_sim_s')} sim-s. Per-episode [episodes.csv](episodes.csv) and JSON include closure XYZ, requested and other block distances, pre-closure approach heights/descent and gripper range. Missing closures remain missing. For lift/placement outcomes, validity, queue starvation and load, use the [main report](../README.md); no near-target geometry alone counts as success.

Low-motion dwell unions half-second windows within a 5 mm TCP box diagonal and a 0.5° gripper range; it can contain wrist rotation or small servo jitter. Strict all-joint instantaneous velocity idle is retained separately. High dwell with zero queue starvation means little translational progress despite available commands. Path length is sensitive to servo jitter. The first ≥2° measured gripper decrease is compared with the nearest requested-colour block centre using production URDF FK and the measured physical base transform. This does not recover finger clearance, jaw orientation or contact.

{chr(10).join(table)}

Training references are seeds 530000–530004 from rtx6000-normal-i3-b00 (two red, two green, one blue), processed with identical FK and closure/dwell definitions. Their layouts differ from evaluation layouts, so this is a descriptive comparison, not a paired intervention. Original metrics and condensed trajectories are retained here; the immutable 10k delivery has the expert videos and detailed training-membership proof. Every evaluation closing distance reproduces the online value within 1 µm, including the absence of closure. The main report's contact sheet and curated videos provide examples for each colour. Video time differs from `/clock` metric time.
'''
(folder/'README.md').write_text(text);print(text[:900])
