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


def cadence(samples):
    return (len(samples)-1)/(samples[-1][0]-samples[0][0]) if len(samples)>1 else 0.


def timing(row):
    frames=row['isaac_frames']['samples']
    return dict(isaac_frame_hz=cadence(frames),
        wrist_hz=cadence(row['transport_arrivals']['wrist']['samples']),
        workspace_hz=cadence(row['transport_arrivals']['workspace']['samples']),
        sim_rtf=(frames[-1][1]-frames[0][1])/(frames[-1][0]-frames[0][0]),
        isaac_major_faults_delta=row['isaac_major_faults_end']-row['isaac_major_faults_start'])


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
              'empty_ticks_after_startup','empty_ticks_pct','first_chunk_s','request_chunk_mean_ms','request_chunk_p95_ms',
              'request_queue_depth_mean','chunk_size_threshold','actions_per_chunk',
              'isaac_frame_hz','wrist_hz','workspace_hz','sim_rtf','isaac_major_faults_delta',
              'closest_blue_center_distance_m','closest_blue','closest_time_s','closing_motion_ge2deg','gripper_min_deg',
              'estimated_min_jaw_gap_m','min_opening_time_s','min_opening_tcp_xyz_m','min_opening_blue_distance_m',
              'any_sustained_lift_24mm','max_any_object_lift_m','distance_basis',
              'rtc_enabled','requested_fps','sim_cap_hz','actions_sent','chunks','max_blue_lift_m','failure_note','video',
              'host_load_start','host_load_end','missed_ticks','empty_ticks_after_startup','nonmonotonic_action_steps']
    with (root/'episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, rows in results.items():
            for row in rows:
                data = dict(row)
                data.update(timing(row))
                sidecar = root/'raw'/name/row['episode']/'characterization.json'
                if sidecar.exists():
                    char=json.loads(sidecar.read_text())
                    data.update({k:char[k] for k in fields if k in char and k not in data})
                    data['failure_note'] += '; FK proxy: '+char['failure_characterization']
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
        '**Status: '+ ('complete (50 scored episodes).' if complete else 'INCOMPLETE; available scored conditions only. Remaining matrix conditions are pending.') + '**', '',
        '| Condition | Success | 95% Wilson interval | Grasped, not placed | No grasp | Mean effective Hz | Mean empty ticks % | Seeds |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    chart = []
    for name in order:
        rows = results.get(name, [])
        if not rows:
            lines.append(f'| {name} | pending | — | — | — | — | — | 900000–900009 |')
            continue
        n = len(rows); k = sum(r['outcome']=='success' for r in rows)
        grasp = sum(r['outcome']=='grasped-not-placed' for r in rows)
        fail = sum(r['outcome']=='no-grasp' for r in rows)
        lo,hi = wilson(k,n)
        hz = statistics.mean(r['effective_action_fps'] for r in rows)
        starvation = statistics.mean(r['empty_ticks_pct'] for r in rows)
        seeds = ', '.join(str(r['seed']) for r in rows)
        lines.append(f'| {name} | {k}/{n} ({k/n:.0%}) | {lo:.1%}–{hi:.1%} | {grasp} | {fail} | {hz:.3f} | {starvation:.2f}% | {seeds} |')
        chart.append((name,k/n,lo,hi,n))
    lines += ['', 'The OLD comparison is a **client-side baseline**: 30 fps, RTC disabled, capture in the client process, normal scheduling; '
        'versus 17.4 Hz, real RTC, isolated capture and FIFO10 dispatch. Both use the current dispatcher implementation; '
        'this does not reproduce the removed pre-1.15 synchronous loop. Sim-side Phase 1 fixes (rgb8 cameras, one frame per sim tick, '
        '17.4 Hz cap, CPU fixes and de-instanced robot) remain in both OLD conditions. '
        'For NEW, sim cap and client fps vary together at 17.4 or 19.4 Hz; other Phase 1 settings remain fixed.', '',
        'OLD versus NEW is a **recipe comparison**, not a pure initialization ablation. The NEW recipe also uses '
        'batch 64, learning rate 1e-4, a frozen VLM/expert-only update, and 20,000 training steps. '
        'Both recipes use the same rec_222812 episodes. Stored configurations also differ in empty-camera count '
        '(OLD 1, NEW 0) and language padding (`longest` versus `max_length`); full differences are in '
        '`provenance/model-config-differences.json`. These differences may affect inference cost and are not isolated here.', '',
        'Scored runs use JointTrajectory topic commands for policy dispatch and discrete FJT actions for homing. '
        'The controllers were restarted to clear retained action status history. Earlier per-tick-FJT runs are '
        'diagnostic only: status arrays grew to thousands of entries and dispatch degraded, including with fresh clients. '
        'The scored matrix follows a consecutive-episode streaming regression; diagnostic traces remain under `diagnostics/`.', '',
        'Validity uses zero missed scheduling slots, no stale-camera aborts and bounded actuator-send cost (p95 below 5 ms). '
        'The effective-Hz band used in the initial dispatcher regression was removed as a scoring validity rule: '
        'an empty action queue is a measured outcome, not a scheduling failure. This rule applies equally to all six conditions. '
        'Mean starvation is the mean per-episode percentage of post-startup opportunities with no queued action; '
        'the denominator is dispatched actions plus empty and missed post-startup ticks. First-chunk latency and '
        'per-chunk request-to-arrival latency/queue depth are retained in the CSV and each `chunks.csv`. '
        'Queue depth is sampled when the observation worker triggers capture; end-to-end latency includes capture, transport, '
        'server queueing and inference. It is not a GPU-only inference duration. Refill threshold remains 0.5 with 50 actions per chunk.', '',
        '## Paired comparisons', '', '| Contrast (B minus A) | Shared seeds | Success difference | Δ effective Hz | Δ empty ticks (pp) | B-only successes | A-only successes | Exact McNemar p |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
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
        dhz = statistics.mean(br[s]['effective_action_fps']-ar[s]['effective_action_fps'] for s in seeds)
        dempty = statistics.mean(br[s]['empty_ticks_pct']-ar[s]['empty_ticks_pct'] for s in seeds)
        prob = min(1.,2*sum(math.comb(d,j) for j in range(min(plus,minus)+1))/2**d) if d else 1.
        lines.append(f'| {cb} − {ca} | {len(seeds)} | {(plus-minus)/len(seeds):+.1%} | {dhz:+.3f} | {dempty:+.2f} | {plus} | {minus} | {prob:.3f} |')
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
        '| Condition | 1-minute CPU load range | Cap Hz | Observed Kit Hz | Wrist / top arrival Hz | Sim RTF |',
        '|---|---:|---:|---:|---:|---:|']
    for name,rows in results.items():
        loads = [r[k][0] for r in rows for k in ('host_load_start','host_load_end') if k in r]
        cfg = configs[name]
        observed=[timing(r) for r in rows]
        means={k:statistics.mean(r[k] for r in observed) for k in observed[0]}
        lines.append(f'| {name} | {min(loads):.2f}–{max(loads):.2f} | {cfg["sim_rate_hz"]} | {means["isaac_frame_hz"]:.2f} | {means["wrist_hz"]:.2f} / {means["workspace_hz"]:.2f} | {means["sim_rtf"]:.3f} |')
    lines += ['', 'Camera/Kit rates are observed over the diagnostic recording window, including brief setup/stop acknowledgement. '
        'A configured ceiling does not guarantee that cadence under shared load. GPU samples are taken before each policy window, not averaged over it. '
        'OLD is `anirudhrani/smolvla_sim_100ep_fft__10ksteps_h200`; NEW is '
        '`/home/aaugus11/w_ckpt/smolvla-base-ft-rec222812-20k-s42-20261113`. Full model/source settings are in each run config and `provenance/`.']
    lines += ['', 'The dry episode uses seed 899999 and is excluded. Its action rate was 17.386 Hz with no grasp; '
        'it exposed the timeline-reset rate-limit trap (60 Hz read-back). Scored runs explicitly apply/read the rate cap '
        'after each scene reset. Full physical scene-bank receipts include pose restore errors and visible-blue checks.', '',
        '## Artifacts', '',
        '- `episodes.csv`: combined raw episode table, including seed, outcome, timings, RTC, cap and failure notes.',
        '- `success-rates.png`: success fractions with 95% Wilson intervals.',
        '- `contact-sheet.jpg`: best/worst episodes by outcome, then final blue-to-cup distance; relative ranks do not imply success.',
        '- `raw/<condition>/`: every scored MP4, pose trace, per-episode result, client/server logs and run config.',
        '- `scene-bank/`: exact shared initial physical poses.', '']
    lines += ['', '## Starvation and shared load', '',
        '| Condition | Mean empty ticks % | Request-to-chunk p95 ms | Median returned-chunk request depth | r(empty %, boundary load) |',
        '|---|---:|---:|---:|---:|']
    possible_load=[]
    for name,rows in results.items():
        traces=[e for r in rows for e in r['chunk_trace']]
        lat=sorted(e['request_to_arrival_s']*1000 for e in traces)
        depths=[e['queue_depth_at_capture_request'] for e in traces]
        x=[(r['host_load_start'][0]+r['host_load_end'][0])/2 for r in rows];y=[r['empty_ticks_pct'] for r in rows]
        mx,my=statistics.mean(x),statistics.mean(y);sx=sum((v-mx)**2 for v in x);sy=sum((v-my)**2 for v in y)
        corr=sum((u-mx)*(v-my) for u,v in zip(x,y))/math.sqrt(sx*sy) if sx and sy else None
        if corr is not None and corr>=.5:possible_load.append(name)
        corr_text=f'{corr:+.2f}' if corr is not None else '—'
        lines.append(f'| {name} | {my:.2f}% | {lat[math.ceil(.95*len(lat))-1]:.0f} | {statistics.median(depths):.1f} | {corr_text} |')
    lines += ['', 'Load correlations are descriptive, with only 5–10 episodes per condition. Boundary loads are one-minute '
        'averages and do not localize sub-second spikes or distinguish low-priority jobs from policy CPU work. '
        'Scene, time and recipe differences also confound attribution. '+
        ('Possible positive load association (r≥0.5): '+', '.join(possible_load)+'.' if possible_load else
         'No condition shows r≥0.5 between starvation and its boundary load samples.'), '',
        'A pre-telemetry NEW run aborted on a 500 ms wrist-camera timeout. Its Kit time window contains no contemporaneous '
        'warning and no MCP call during the policy window; no frame/fault trace existed then, so the cause remains unlocalized. '
        'That run, earlier action-history diagnostics, untraced runs and the telemetry validation are archived separately. '
        'The final traced matrix retains the 500 ms deadline and excludes those diagnostic attempts.', '']
    characterization=root/'failure-characterization.json'
    if characterization.exists():
        chars=json.loads(characterization.read_text())
        if a.complete and len(chars)!=50:raise RuntimeError('Characterization incomplete')
        lines += ['## Failure characterization', '',
            'Per-episode `failure-characterization.csv` and `tcp_analysis.csv` use the production URDF `tcp_link` FK '
            'on the recorded mock-controller reference. These are commanded fixed-jaw-tip-to-blue-centre distance proxies, '
            'not historical measured physical TCP trajectories. Two static Isaac checks found URDF-vs-physical tip errors '
            '0.042–0.050 mm and command-vs-physical offsets about 0.326 mm; these do not bound dynamic tracking lag. '
            'Closing motion means a ≥2° decrease from a preceding opening. Estimated jaw gap uses the production linear '
            'calibration and does not establish finger contact or a completed grasp. Closure/min-opening times, positions '
            'and nearest objects are retained per episode.', '',
            '| Condition | Closest blue-centre FK distance, median (range) mm | Closing motion | Any sustained object lift |',
            '|---|---:|---:|---:|']
        for name in results:
            rr=[r for r in chars if r['condition']==name];ds=[1000*r['closest_blue_center_distance_m'] for r in rr]
            lines.append(f'| {name} | {statistics.median(ds):.1f} ({min(ds):.1f}–{max(ds):.1f}) | {sum(r["closing_motion_ge2deg"] for r in rr)}/{len(rr)} | {sum(r["any_sustained_lift_24mm"] for r in rr)}/{len(rr)} |')
        lines += ['', '`new-episode-sequences.jpg` shows four video frames for the same scene seed 900003 in each NEW condition. '
            'Video time includes brief episode setup/stop acknowledgement; it is not an exact action-event alignment.', '']
    parity=root/'camera-parity/summary.json'
    if parity.exists():
        lines += ['## Visual parity at recorded home', '',
            '`camera-parity/comparison.png` compares both live feeds with rec_222812 episode 0 frame 0 at its exact recorded '
            'commanded home state. Both are RGB 640×480. Fixed robot/cup features align closely by eye; no gross channel swap, '
            'camera rotation/FOV shift or resolution mismatch is apparent. Live static regions are modestly brighter: table '
            'median RGB [83,104,115] versus dataset [74,93,103], with common robot/cup regions differing by about 10–14 channel '
            'levels. This does not isolate lighting, tone mapping or codec/range effects. Lego layouts differ because the dataset has no '
            'preserved seed/body-pose receipt. Static ROI appearance metrics, current camera metadata and the FK validation '
            'are in `camera-parity/summary.json` and its README. This is not whole-frame pixel parity or a historical extrinsics calibration.', '']
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
            outcome_rank = {'success': 0, 'grasped-not-placed': 1, 'no-grasp': 2}
            ranked = sorted(rows,key=lambda r:(outcome_rank[r['outcome']],r.get('final',{}).get('r_xy_m',float('inf'))))
            for label,row in [('best',ranked[0]),('worst',ranked[-1])]:
                source=root/'raw'/name/row['episode']/'viewport.mp4'
                if not source.exists():continue
                frame=root/f'.frame-{name}-{label}.png'
                t = max(0,min((row.get('time_to_place_s') or row.get('time_to_grasp_s') or row['time_s']*.65)+.8,row['time_s']-1))
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
