#!/usr/bin/env python3
"""Build a local evaluation report from the episode harness's raw results."""
import argparse
import csv
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    d = 1 + z*z/n
    c = (k/n + z*z/(2*n))/d
    h = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--complete', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    order = ['old-before', 'old-after', 'new-17.4-rtc-on', 'new-17.4-rtc-off', 'new-19.4-rtc-on', 'new-19.4-rtc-off']
    results, configs = {}, {}
    for name in order:
        path = root/'raw'/name
        if (path/'results.jsonl').exists():
            results[name] = [json.loads(line) for line in (path/'results.jsonl').read_text().splitlines() if line]
            configs[name] = json.loads((path/'run_config.json').read_text())
            rows = results[name]
            if len({r['seed'] for r in rows}) != len(rows):
                raise RuntimeError(f'Duplicate scored seed in {name}')
            for row in rows:
                if row['outcome'] not in {'success','grasped-not-placed','no-grasp'}:
                    raise RuntimeError(f'Invalid episode in {name}: {row["episode"]}')
                cap = row.get('sim_cap_hz', row.get('sim_timing',{}).get('/app/runLoops/main/rateLimitFrequency'))
                if cap is None or abs(cap-configs[name]['sim_rate_hz']) > 1e-4:
                    raise RuntimeError(f'Sim cap mismatch in {name}: {row["episode"]}')
    expected = {name: 5 if name.startswith('old') else 10 for name in order}
    complete = all(len(results.get(name, [])) == n for name,n in expected.items())
    if a.complete and not complete:
        raise RuntimeError('Cannot mark report complete: evaluation matrix is incomplete')
    fields = ['condition','episode','seed','outcome','time_to_grasp_s','time_to_place_s','time_s','effective_action_fps',
              'rtc_enabled','requested_fps','sim_cap_hz','actions_sent','chunks','max_blue_lift_m','failure_note','video',
              'host_load_start','host_load_end','missed_ticks','empty_ticks_after_startup','nonmonotonic_action_steps']
    with (root/'episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, rows in results.items():
            for row in rows:
                data = dict(row)
                data['sim_cap_hz'] = row.get('sim_cap_hz',row.get('sim_timing',{}).get('/app/runLoops/main/rateLimitFrequency'))
                data['video'] = f'raw/{name}/{row["episode"]}/viewport.mp4'
                data['missed_ticks'] = row.get('client',{}).get('missed_action_ticks_after_startup')
                data['empty_ticks_after_startup'] = row.get('client',{}).get('empty_action_ticks_after_startup')
                data['nonmonotonic_action_steps'] = row.get('client',{}).get('nonmonotonic_action_steps')
                writer.writerow(data)
    lines = ['# SO-ARM101 blue-lego evaluation', '',
        'This evaluation measures whether the OLD random-init SmolVLA checkpoint and the NEW checkpoint fine-tuned from '
        '`lerobot/smolvla_base` pick a blue lego and place it in the blue cup in the Isaac twin. '
        'Scenes are paired by saved physical block/cup poses, the arm is homed before every episode, and each policy window lasts up to 45 wall seconds. '
        'The task is "Pick a blue lego and place it in blue cup"; state is the rec_222812-compatible `/joint_states` controller reference. '
        'Videos, pose traces, resolved settings and model/fork commits are retained under `raw/`; `episodes.csv` combines the per-episode measurements.', '',
        '**Status: '+ ('complete (50 scored episodes).' if complete else 'INCOMPLETE; preparation/partial results only. NEW runs await the lead’s checkpoint-ready message.') , '',
        '| Condition | Success | 95% Wilson interval | Grasped, not placed | No grasp | Mean effective Hz | Seeds |',
        '|---|---:|---:|---:|---:|---:|---|']
    chart = []
    for name in order:
        rows = results.get(name, [])
        if not rows:
            lines.append(f'| {name} | pending | — | — | — | — | 900000–900009 |')
            continue
        n = len(rows); k = sum(r['outcome']=='success' for r in rows)
        grasp = sum(r['outcome']=='grasped-not-placed' for r in rows)
        fail = sum(r['outcome']=='no-grasp' for r in rows)
        lo,hi = wilson(k,n)
        hz = statistics.mean(r['effective_action_fps'] for r in rows)
        seeds = ', '.join(str(r['seed']) for r in rows)
        lines.append(f'| {name} | {k}/{n} ({k/n:.0%}) | {lo:.1%}–{hi:.1%} | {grasp} | {fail} | {hz:.3f} | {seeds} |')
        chart.append((name,k/n,lo,hi,n))
    lines += ['', 'The OLD comparison is a **client-side baseline**: 30 fps, RTC disabled, capture in the client process, normal scheduling; '
        'versus 17.4 Hz, real RTC, isolated capture and FIFO10 dispatch. Both use the current dispatcher implementation; '
        'this does not reproduce the removed pre-1.15 synchronous loop. Sim-side Phase 1 fixes (rgb8 cameras, one frame per sim tick, '
        '17.4 Hz cap, CPU fixes and de-instanced robot) remain in both OLD conditions. '
        'For NEW, sim cap and client fps vary together at 17.4 or 19.4 Hz; other Phase 1 settings remain fixed.', '',
        'OLD versus NEW is a **recipe comparison**, not a pure initialization ablation. The NEW recipe also uses '
        'batch 64, learning rate 1e-4, a frozen VLM/expert-only update, and 20,000 training steps. '
        'Both recipes use the same rec_222812 episodes.', '',
        'Scored runs use JointTrajectory topic commands for policy dispatch and discrete FJT actions for homing. '
        'The controllers were restarted to clear retained action status history. Earlier per-tick-FJT runs are '
        'diagnostic only: status arrays grew to thousands of entries and dispatch degraded, including with fresh clients. '
        'The scored matrix follows a consecutive-episode streaming regression; diagnostic traces remain under `diagnostics/`.', '',
        '## Paired comparisons', '', '| Contrast (B minus A) | Shared seeds | Success difference | B-only successes | A-only successes | Exact McNemar p |',
        '|---|---:|---:|---:|---:|---:|']
    pairs = [('old-before','old-after')]
    new = [name for name in order if name.startswith('new')]
    pairs.extend(itertools.combinations(new,2))
    pairs.extend(('old-after',name) for name in new)
    for ca,cb in pairs:
        if ca not in results or cb not in results:
            continue
        ar = {r['seed']:r for r in results[ca]}; br = {r['seed']:r for r in results[cb]}
        seeds = sorted(ar.keys() & br.keys())
        plus = sum(br[s]['outcome']=='success' and ar[s]['outcome']!='success' for s in seeds)
        minus = sum(ar[s]['outcome']=='success' and br[s]['outcome']!='success' for s in seeds)
        d = plus+minus
        prob = min(1.,2*sum(math.comb(d,j) for j in range(min(plus,minus)+1))/2**d) if d else 1.
        lines.append(f'| {cb} − {ca} | {len(seeds)} | {(plus-minus)/len(seeds):+.1%} | {plus} | {minus} | {prob:.3f} |')
    lines += ['', 'Differences use only shared scene seeds. Wilson intervals describe each condition’s binomial success rate; '
        'they are not intervals on the paired difference. With only 5 or 10 episodes, zero observed successes does not establish '
        'zero underlying success probability, and a small/no paired difference does not establish equivalence. '
        'Model sampling and physics are stochastic; shared poses pair the initial scene, not every source of randomness.', '',
        '## Scoring and timing limits', '',
        'The 1.8 scorer uses `/objects_poses_sim` and `/drop_poses`. A grasp is proxied by a blue block rising at least 24 mm '
        'for 0.5 wall seconds. Success requires a sustained lift and a blue block centre inside the blue cup’s conservative '
        '33 mm radius interior, with floor/rim clearance, for two wall seconds and still inside at stop. This is a centre-based '
        'geometric proxy; the scorer does not establish full-block containment, finger contact or release. '
        'Grasp/place times record the onset of the qualifying sustained interval, relative to episode start; blank means not observed. '
        'Video/pose review is needed for any apparent success. Effective Hz excludes startup wait and uses first-to-last dispatch timing. '
        'A missing/stale stream or setup error invalidates an episode; such diagnostic attempts are excluded from the scored matrix.', '',
        '## Host load and provenance', '',
        'Isaac runs on GPU 0, policy on GPU 1 (pinned by UUID); 28 logical CPUs. The shared host was not assumed idle. '
        'The harness records CPU load before/after each episode and per-episode GPU utilization/memory snapshots. '
        'The collector has a separate domain/instance and was coordinated to remain idle. No unrelated processes were stopped.', '',
        '| Condition | 1-minute load range (episode boundaries) | Client / sim Hz | RTC | Checkpoint |',
        '|---|---:|---:|---|---|']
    for name,rows in results.items():
        loads = [r[k][0] for r in rows for k in ('host_load_start','host_load_end') if k in r]
        cfg = configs[name]
        lines.append(f'| {name} | {min(loads):.2f}–{max(loads):.2f} | {cfg["fps"]} / {cfg.get("sim_rate_hz",17.4)} | {cfg["rtc_enabled"]} | `{cfg["pretrained"]}` |')
    lines += ['', 'The dry episode uses seed 899999 and is excluded. Its action rate was 17.386 Hz with no grasp; '
        'it exposed the timeline-reset rate-limit trap (60 Hz read-back). Scored runs explicitly apply/read the rate cap '
        'after each scene reset. Full physical scene-bank receipts include pose restore errors and visible-blue checks.', '',
        '## Artifacts', '',
        '- `episodes.csv`: combined raw episode table, including seed, outcome, timings, RTC, cap and failure notes.',
        '- `success-rates.png`: success fractions with 95% Wilson intervals.',
        '- `contact-sheet.jpg`: closest/furthest-to-cup episodes per available condition; these are relative rankings, not success claims.',
        '- `raw/<condition>/`: every scored MP4, pose trace, per-episode result, client/server logs and run config.',
        '- `scene-bank/`: exact shared initial physical poses.', '']
    (root/'README.md').write_text('\n'.join(lines))
    if chart:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(figsize=(11,5.5))
        for i,(name,rate,lo,hi,n) in enumerate(chart):
            ax.bar(i,rate,color='#3173b8' if name.startswith('old') else '#39866b',width=.6)
            ax.errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],color='#24384a',capsize=6,fmt='none',lw=2)
            ax.text(i,min(1.05,hi+.035),f'{rate:.0%} (n={n})',ha='center',fontsize=10)
        ax.set(ylim=(0,1.12),ylabel='Success rate',title='Blue lego → blue cup · 95% Wilson intervals',xticks=range(len(chart)),xticklabels=[r[0] for r in chart])
        ax.tick_params(axis='x',rotation=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(root/'success-rates.png',dpi=180);plt.close(fig)
        from PIL import Image, ImageDraw, ImageFont
        tiles=[]
        font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14)
        for name,rows in results.items():
            ranked = sorted(rows,key=lambda r:r.get('final',{}).get('r_xy_m',float('inf')))
            for label,row in [('closest',ranked[0]),('furthest',ranked[-1])]:
                source=root/'raw'/name/row['episode']/'viewport.mp4'
                if not source.exists():continue
                frame=root/f'.frame-{name}-{label}.png'
                t = max(0,min(row.get('time_to_grasp_s') or row['time_s']*.65,row['time_s']-1))
                subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',str(source),'-frames:v','1',str(frame)],check=True)
                im=Image.open(frame).convert('RGB');im.thumbnail((480,300))
                tile=Image.new('RGB',(500,360),'#f3f5f7');tile.paste(im,((500-im.width)//2,0))
                draw=ImageDraw.Draw(tile)
                draw.text((10,305),f'{name} · {label}',fill='#203348',font=font)
                draw.text((10,327),f'{row["episode"]} seed {row["seed"]} {row["outcome"]}',fill='#203348',font=font)
                tiles.append(tile);frame.unlink()
        if tiles:
            sheet=Image.new('RGB',(1000,360*math.ceil(len(tiles)/2)),'white')
            for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*500,(i//2)*360))
            sheet.save(root/'contact-sheet.jpg',quality=90)
    print(json.dumps(dict(complete=complete,episodes=sum(map(len,results.values())),conditions=list(results))))

if __name__=='__main__':main()
