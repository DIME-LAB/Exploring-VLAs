"""Write one completed checkpoint/RTC-arm report from its audited summaries."""
from pathlib import Path
import argparse,json
from checkpoint_identity import checkpoint_label

p=argparse.ArgumentParser();p.add_argument('root',type=Path);root=p.parse_args().root
assert not (root/'provenance/delivery-manifest.json').exists(),'Refusing to change a finalized delivery'
s=json.loads((root/'summary.json').read_text());o=s['overall'];contract=s['contract']
rows=json.loads((root/'episodes.json').read_text());step=int(Path(contract['pretrained']).parent.name);label=checkpoint_label(contract['pretrained'])
ledger=json.loads((root/'raw/window-ledger.json').read_text());pre=json.loads((root/'raw/preflight.json').read_text())
rtc=contract['rtc_enabled'];arm='on' if rtc else 'off';suite='main' if rtc else 'main-rtc-off'
def rate(g,k):return f"{g[k]['k']}/{g[k]['n']}"
def value(g,k,scale=1):
    x=g[k]['mean'];return '—' if x is None else f'{scale*x:.2f}'
def ci(g):return '–'.join(f'{100*x:.1f}%' for x in g['success']['wilson95'])
def fmt(x):return '—' if x is None else f'{x:.2f}'
table=['| Set | Success (95% Wilson) | Target grasp | Correct cup | Wrong-block lift | Target wrong cup | Wrong-block placement | Invalid / starved | Closing distance mean (mm) | Empty ticks mean |','|---|---|---|---|---|---|---|---|---|---|']
for name,g in [('Overall',o)]+[(c.title(),s['per_color'][c]) for c in ('red','green','blue')]:
    table.append(f"| {name} | {rate(g,'success')} ({ci(g)}) | {rate(g,'grasp_success')} | {rate(g,'correct_cup')} | {rate(g,'wrong_color_lift')} | {rate(g,'target_wrong_cup')} | {rate(g,'wrong_block_placement')} | {g['invalid']} / {g['starved']} | {value(g,'closing_distance_m',1000)} | {value(g,'empty_ticks_pct')}% |")
invalid=[]
for r in rows:
    if r['reported_outcome']=='invalid':invalid.append(f"Seed {r['seed']}: {r['validity_reasons']}; observed physical outcome {r.get('outcome','missing')}.")
selector='The strict frozen best-checkpoint selector requires 30 valid MAIN rows; this set is ineligible for that selector.' if rtc and o['invalid'] else ''
media=json.loads((root/'video-verification.json').read_text())
gallery='\n'.join(f"- [{m['color']} seed {m['seed']}: {m['selection']}]({m['file']})" for m in media)
perseed=['| Seed | Colour | Outcome | Valid | Closing / closest (mm) | Grasp / place (sim s) | Sim Hz | Empty ticks (%) |','|---|---|---|---|---|---|---|---|']
for r in rows:
    distance=lambda k:fmt(1000*r[k]) if r.get(k) is not None else '—'
    perseed.append(f"| {r['seed']} | {r['color']} | {r['reported_outcome']} | {r['valid']} | {distance('closing_distance_m')} / {distance('closest_target_distance_m')} | {fmt(r.get('time_to_grasp_s'))} / {fmt(r.get('time_to_place_s'))} | {fmt(r.get('effective_action_fps'))} | {fmt(r.get('empty_ticks_pct'))} |")
loads=[r[k] for r in rows for k in ('load1_start','load1_end') if r.get(k) is not None]
guidance='RTC execution horizon 10, maximum guidance 10, EXP schedule, and inference delay measured in simulation time' if rtc else 'RTC guidance disabled, with native plain chunk execution and native overlap aggregation retained'
text=f'''Checkpoint **phase3-{label} scored {rate(o,'success')} confirmed successes** (95% Wilson {ci(o)}) on the 30 held-out MAIN scenes, RTC {arm}. Target grasp: {rate(o,'grasp_success')}; correct cup: {rate(o,'correct_cup')}; invalid: {o['invalid']}; starved: {o['starved']}; not run: {o['not_run']}. This report covers one checkpoint and one deployment arm; the parent report links companion windows and CPU diagnostics.

{chr(10).join(table)}

Invalid and non-run scenes remain in the planned denominator and cannot contribute confirmed success. Valid-only success is {o['valid_only_success']['k']}/{o['valid_only_success']['n']}; its Wilson interval is in summary.json. {selector} {' '.join(invalid)} Starvation is an overlapping outcome, not an exclusion. No attempted scene was silently retried. Correct-cup rate above uses all planned scenes; conditional on placing a target block it is {o['correct_cup_given_target_placement']['k']}/{o['correct_cup_given_target_placement']['n']} (undefined if the denominator is zero). Ten scenes per colour leave wide uncertainty. Wilson intervals are marginal; paired differences in companion reports are descriptive.

![Success and Wilson intervals](success-rate.png)

Deployment: client and server action labels at 30 Hz **simulation time**, re-observation after ten executed actions, 50-action chunks, {guidance}. Ten actions is one third of a sim-second and permits visual feedback while retaining overlap. The qualified ten-empty-slot retry is unchanged. Native refill threshold is 0.8; the adapter's ten-action cadence determines requests. Eager inference is fixed across checkpoints. One persistent client was warmed with 100 non-actuating actions before scoring. Warmup contributes no episodes. The production camera, normalization and model-RPC paths use top then wrist, RGB 640×480, no rename/empty camera, and measured `/joint_states_measured`. The arm is homed and the sealed layout restored before each episode; maximum duration is 45 sim-seconds.

Validity requires zero missed simulation-time slots after startup, no stale-clock/camera or model/RPC errors, nondecreasing action indices, send p95 below 5 ms, cap read-back 30 Hz after reset, and the fixed observation cadence. {'Guided prefixes are required for RTC-on.' if rtc else 'The configured RTC-off mode is checked.'} Effective Hz and starvation remain outcomes. Mean dispatch was {value(o,'effective_action_fps')} sim-Hz / {value(o,'effective_wall_action_fps')} wall-Hz; mean empty ticks {value(o,'empty_ticks_pct')}%; worst episode send p95 {o['send_p95_ms']['max']:.3f} ms. Dispatch jitter is measured against received `/clock` ticks, not sub-tick hardware timing. Individual dispatch/request/chunk/queue traces, receiver timing and inference delays remain in raw results.

Window: {ledger['started_utc']} to {ledger.get('stopped_utc','NOT STOPPED')}. Mean scored action duration was {value(o,'wall_time_s')} wall-seconds; including setup, {value(o,'total_episode_wall_s')} wall-seconds. Isaac and policy used GPU 0, domain 92, loopback-only CycloneDDS; GPU 1 and training-box GPUs were untouched. Managed cleanup exit code: {ledger.get('cleanup_exit_code')}. Boundary one-minute host load ranged {min(loads):.2f}–{max(loads):.2f}; these are scene start/end samples, not a continuous load trace. Per-episode load and Isaac major-fault deltas are in the CSV. {'With no starvation, this sample cannot support a starvation/load association.' if not o['starved'] else 'Load was not randomized; compare the per-seed load and starvation fields before attributing gaps.'}

The first-closing metric detects ≥2° measured gripper closing from its preceding maximum and measures the fixed-jaw URDF TCP distance to the nearest requested-colour block centre in world coordinates. It is a geometric proxy, not finger-surface clearance, yaw alignment or contact. Mean closing distance: {value(o,'closing_distance_m',1000)} mm from {o['closing_distance_m']['n']} available episodes; mean closest approach: {value(o,'closest_target_distance_m',1000)} mm. An episode that never closes has a missing closing distance. Summaries retain all observed diagnostic values and separate valid-only counts/means. Target grasp requires a ≥24 mm lift sustained for ≥0.5 sim-s; correct placement requires whole-block matching-cup containment/settling for ≥0.5 sim-s.

![Distances and starvation](distances-starvation.png)

![Selected episode contact sheet](contact-sheet.png)

{gallery}

{'No success video exists in this set. Selected failures cover the closest/farthest available valid closing distances by colour and an invalid example when present.' if not o['success']['k'] else 'Success and failure examples are included where available.'} Selected MP4s were fully decoded and hashed in [video-verification.json](video-verification.json). Video frame time differs from episode `/clock` time. All original movies remain under the remote run directory.

{chr(10).join(perseed)}

[episodes.csv](episodes.csv) and [summary.json](summary.json) contain all planned scenes, counts, intervals, source contract and telemetry summaries. [raw](raw/) includes result/config/process/warmup/health receipts; [raw-traces.tar.gz](raw-traces.tar.gz) preserves all physical and pose streams. [videos](videos/) contains the curated MP4s. Reproduce summary/CSV with `PYTHONPATH=code python3 code/report_scored_window.py raw --step {step} --suite {suite} --out /tmp/phase3-{label}-rtc-{arm}`. Delivered code also renders figures and verifies raw traces/videos.

Checkpoint: `{contract['pretrained']}`. Weights SHA256: `{pre['checkpoint']['model_sha256']}`. Bank SHA256: `3b3569389d3301cc26fc0b9ed892c032888b2b4531f97156296f20f079648387`. Functional source `c9199c3`, readiness `6c465ef`, fork `e1e201098b2056827908e76f58bbf5733beb4d82`; all twelve qualified hashes are in preflight.json. Seeds are exactly 900000–900029, colour = (red, green, blue)[seed % 3], ten per colour, paired across windows. MAIN policy-flow RNG was not paired. These exact task strings do not establish paraphrase generalization. Remote run: `{ledger['root']}/`.
'''
if (root/'behaviour/episodes.json').exists():
    bh=json.loads((root/'behaviour/episodes.json').read_text());closed=sum(r.get('ever_closed_ge2deg') is True for r in bh);near=sum(r.get('target_nearer_at_closing') is True for r in bh)
    text+=f'\n[Behavioural metrics](behaviour/episodes.csv) report target/other approach distances, closure XYZ/time, descent and low-motion dwell: closed≥2° in {closed}/30, with the requested colour nearest at closure in {near}/30. Missing closures are retained.\n'
if (root/'comparison/README.md').exists():text+='\n[Paired comparison](comparison/README.md) reports shared-seed differences and their limits.\n'
if (root/'behaviour/colour-swap-README.md').exists():text+='\n[CPU HOME colour-word probe](behaviour/colour-swap-README.md) reports saved tokens and open-loop predictions.\n'
(root/'README.md').write_text(text);print(text[:800])
