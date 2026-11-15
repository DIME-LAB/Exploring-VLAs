"""Replay training labels or executed commands through the collector's native handshake."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,threading,time
import numpy as np
REPO=Path(__file__).resolve().parents[3]
from native_environment import qualified_host
sys.path.insert(0,str(REPO/'linux-env/scripts/collect'))
sys.path.insert(0,str(REPO/'omni-kit-mcp'))
from collect import mcp
from held_tick import CollectorTransport,float32
from phase3_metrics import PhysicalScore

CUT='bf7336d93d5ba054c5edc66cea685725eb30eb40'
LOOPBACK_SHA256='d1247d7edefc8f507f43ff15c129aeb5754e70bd7fd34bf4022966c879482a91'

def require_environment():
 assert os.environ.get('ROS_DOMAIN_ID')=='92' and os.environ.get('SOARM_KIT_MCP_PORT')=='8781'
 assert os.environ.get('RMW_IMPLEMENTATION')=='rmw_cyclonedds_cpp'
 assert os.environ.get('CYCLONEDDS_URI')==f'file://{REPO}/linux-env/cyclonedds.xml'
 assert hashlib.sha256((REPO/'linux-env/cyclonedds.xml').read_bytes()).hexdigest()==LOOPBACK_SHA256
 assert all(key not in os.environ for key in ('ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_STATIC_PEERS'))
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Replay client must remain CPU-only'


def validate_collector_cut():
 import subprocess
 assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==CUT
 core={}
 for name in ('collection_runtime.py','protocol.py','collect.py','frame_join.py','tick_contract.py','image_transport.py','image_take.c','contract.py','success.py','pose_math.py','cup_geometry.py'):
  path=REPO/'linux-env/scripts/collect'/name;original=subprocess.check_output(['git','-C',str(REPO),'show',CUT+':linux-env/scripts/collect/'+name]);assert path.read_bytes()==original,'Collector cut differs: '+name;core[name]=hashlib.sha256(original).hexdigest()
 return core


def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--fixture-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--commands',choices=('clean','executed'),required=True);a=p.parse_args()
 hostroot=Path(qualified_host(REPO)['root']);assert a.out.resolve().is_relative_to(hostroot) and not a.out.exists();a.out.mkdir(parents=True)
 require_environment()
 core=validate_collector_cut()
 raw=a.fixture.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.fixture_sha256;fixture=json.loads(raw);assert fixture['kind']=='native-held-replay-v1' and len(fixture['episodes'])==3
 fkpath=REPO/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/compute_workspace.py';spec=importlib.util.spec_from_file_location('eval_native_fk',fkpath);fk=importlib.util.module_from_spec(spec);spec.loader.exec_module(fk)
 transport=None;recording=False;results=[]
 try:
  transport=CollectorTransport()
  mcp('run_python',{'code':(REPO/'linux-env/scripts/collect/configure_clock.py').read_text()+'\nresult=configure(30)'})
  installed=mcp('run_python',{'code':f'import sys\nsys.path.insert(0,{str(REPO/"linux-env/scripts/collect")!r})\nimport collection_runtime as c\nresult=c.install("collect")'})
  assert installed['loaded_sha256']==core['collection_runtime.py'] and installed['physics_config']['dynamics']=='cpu' and installed['articulation_audit']['fixed_base'] is True
  assert installed['cuda_device_count']==1 and installed['cuda_device']==0 and installed['renderer_multi_gpu'] is False
  transport.generation.install(installed)
  deadline=time.monotonic()+20
  while transport.publisher.get_subscription_count()<1:
   if time.monotonic()>deadline:raise TimeoutError('Collector command graph discovery')
   threading.Event().wait(.05)
  mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'});transport.session.read()
  # The collector naturally holds after each command ACK / one-frame advance.
  for index,episode in enumerate(fixture['episodes']):
   out=a.out/f'ep{index:02d}';out.mkdir();snapshots=[];steps=[];began=time.monotonic()
   held=mcp('run_python',{'code':'import collection_runtime as c\nr=c.runtime()\nresult=dict(held=not r.timeline.is_playing(),busy=r.busy,failure=r.failure)'});assert held==dict(held=True,busy=False,failure=None)
   initial=episode['initial'];request=dict(state=initial['state'],drive_targets=initial['drive_targets'],bodies=initial['bodies'])
   restored=mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().restore('+repr(request)+',arm=True,restore_drives=True)'})
   transport.generation.install(restored)
   transport.sensors.prune_durable_camera_history()
   mcp('run_python',{'code':'import collection_runtime as c\nresult=c.runtime().advance()'})
   current=transport.session.read();snapshots.append(current)
   assert max(abs(x-y) for x,y in zip(current['robot_base_world_pose']['position'],initial['robot_base_world_pose']['position']))<1e-6
   scorer=PhysicalScore(episode['color'],current,fk);scorer.update(current)
   mcp('soarm101.start_recording',{'file_name':str(out/'viewport_capture.mp4')});recording=True
   actions=episode['clean_actions_rad' if a.commands=='clean' else 'executed_actions_rad']
   for action_index,action in enumerate(actions):
    proof,current=transport.session.step(action);steps.append(dict(action_index=action_index,**proof));snapshots.append(current);scorer.update(current)
   # Finish on the final recorded tick; no unrecorded hold steps for scoring.
   video=mcp('soarm101.stop_recording');recording=False
   q=np.asarray([s['state'] for s in snapshots[:-1]]);reference=np.asarray(episode['recorded_states_rad']);error=np.degrees(q-reference)
   start_q=episode['recorded_states_rad'][0][5];max_grip=start_q;closure=len(actions)
   for i,ref in enumerate(episode['recorded_states_rad']):
    max_grip=max(max_grip,ref[5])
    if max_grip-ref[5]>=math.radians(2):closure=i;break
   def stats(e):return dict(rms_deg=np.sqrt(np.mean(e*e,axis=0)).tolist(),max_abs_deg=np.max(np.abs(e),axis=0).tolist(),frames=len(e)) if len(e) else None
   tcp=fk.forward_kinematics_batch(q[:,:5]);tcpref=fk.forward_kinematics_batch(reference[:,:5]);tcp_error=np.linalg.norm(tcp-tcpref,axis=1)
   result=dict(seed=episode['seed'],color=episode['color'],commands=a.commands,clock='collector native held tick',action_count=len(actions),wall_s=time.monotonic()-began,fixture_sha256=a.fixture_sha256,source_frames_sha256=episode['source_sha256'],loopback_xml_sha256=LOOPBACK_SHA256,collector_cut=CUT,collector_source_sha256=core,installed=installed,reset=restored,unrecorded_reset_bootstrap_frames=1,reset_velocity_semantics='Collector restore zeros velocities; original momentum not resumed',state_tracking=stats(error),preclosure_state_tracking=stats(error[:closure]),tcp_tracking=dict(rms_mm=float(np.sqrt(np.mean(tcp_error*tcp_error))*1000),max_mm=float(np.max(tcp_error)*1000),preclosure_rms_mm=float(np.sqrt(np.mean(tcp_error[:closure]**2))*1000),preclosure_max_mm=float(np.max(tcp_error[:closure])*1000)),clean_executed_max_delta_deg=episode['clean_executed_max_delta_deg'],video=video,**scorer.result())
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'physical.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in snapshots));(out/'commands.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps));results.append(result);print(json.dumps(dict(event='native-replay',seed=result['seed'],commands=a.commands,grasp=result['grasp_success'],success=result['success'],preclosure_tcp_rms_mm=result['tcp_tracking']['preclosure_rms_mm'])),flush=True)
  (a.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
 finally:
  if recording:
   try:mcp('soarm101.stop_recording')
   except Exception:pass
  if transport is not None:
   retired=transport.close();(a.out/'retired.json').write_text(json.dumps(retired,indent=2)+'\n')
if __name__=='__main__':main()
