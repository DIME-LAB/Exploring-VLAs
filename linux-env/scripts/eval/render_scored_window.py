#!/usr/bin/env python3
"""Render outcome figures and selected videos from a completed scored window."""
from pathlib import Path
import argparse,csv,json,math,statistics,subprocess,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from report_scored_window import select_examples
from checkpoint_identity import checkpoint_label as model_label

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
assert not (root/'provenance/delivery-manifest.json').exists(),'Refusing to change a finalized delivery'
rows=json.loads((root/'episodes.json').read_text());summary=json.loads((root/'summary.json').read_text());overall=summary['overall'];checkpoint_label=model_label(summary['contract']['pretrained']);colors=('red','green','blue');rtc_label='on' if summary['contract']['rtc_enabled'] else 'off'
fig,ax=plt.subplots(figsize=(8,4.4));groups=[('Overall',overall)]+[(c.title(),summary['per_color'][c]) for c in colors]
for i,(label,g) in enumerate(groups):
 s=g['success'];lo,hi=s['wilson95'];v=100*s['rate'];ax.bar(i,v,color=['#576574','#c4473d','#388d65','#3c78b1'][i],width=.55)
 ax.errorbar(i,v,yerr=[[v-100*lo],[100*hi-v]],color='#333333',fmt='o',capsize=6)
 ax.text(i,100*hi+3,f"{s['k']}/{s['n']}",ha='center')
ax.set_xticks(range(len(groups)),[g[0] for g in groups]);ax.set(ylim=(0,100),ylabel='Confirmed success (%)',title=f'Phase3 {checkpoint_label} · MAIN exact task · RTC {rtc_label}\n95% Wilson intervals; all planned scenes included');ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(root/'success-rate.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
for color,paint in zip(colors,('#c4473d','#388d65','#3c78b1')):
 rs=[r for r in rows if r['color']==color];x=[r['seed']-900000 for r in rs]
 axs[0].scatter(x,[1000*r['closing_distance_m'] if r.get('closing_distance_m') is not None else np.nan for r in rs],label=color,color=paint,s=35)
 axs[1].scatter(x,[r.get('empty_ticks_pct',np.nan) for r in rs],color=paint,s=35)
 axs[2].scatter(x,[r.get('missed_sim_slots',np.nan) for r in rs],color=paint,s=35)
axs[0].set(ylabel='First-closing distance (mm)',title='Per-seed physical failure metric and queue starvation');axs[0].legend(frameon=False,ncol=3)
axs[1].set(ylabel='Empty ticks after startup (%)')
axs[2].set(xlabel='Held-out seed minus 900000',ylabel='Missed sim slots');axs[2].set_xticks(range(0,30,3))
for ax in axs:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(root/'distances-starvation.png',dpi=180);plt.close(fig)
selected=select_examples(rows)
(root/'videos').mkdir(exist_ok=True);(root/'frames').mkdir(exist_ok=True)
media=[];fig,axs=plt.subplots(len(selected),3,figsize=(12,3.15*len(selected)),squeeze=False)
for i,(r,why) in enumerate(selected):
 movie=root/'videos'/f"seed-{r['seed']}.mp4"
 if not movie.exists():raise FileNotFoundError(f'Fetch selected video to {movie}')
 meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames:format=duration','-of','json',str(movie)],text=True));duration=float(meta['format']['duration'])
 subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True,capture_output=True)
 media.append(dict(seed=r['seed'],color=r['color'],selection=why,file=str(movie.relative_to(root)),duration_s=duration,frames=int(meta['streams'][0]['nb_read_frames']),sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),fully_decoded=True))
 for j,t in enumerate((1,duration*.5,max(1,duration-2))):
  frame=root/'frames'/f"seed-{r['seed']}-{j}.png";subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(movie),'-frames:v','1',str(frame)],check=True)
  axs[i,j].imshow(Image.open(frame));axs[i,j].axis('off');axs[i,j].set_title(f"{r['color']} · seed {r['seed']} · video {t:.1f}s",fontsize=9)
 axs[i,0].text(0,-.07,why,transform=axs[i,0].transAxes,fontsize=9)
fig.suptitle('Selected scored episodes · start / middle / end\nVideo time is frame time; timing metrics use ROS /clock.',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(root/'contact-sheet.png',dpi=140);plt.close(fig)
(root/'video-verification.json').write_text(json.dumps(media,indent=2)+'\n')
print(json.dumps(media,indent=2))
