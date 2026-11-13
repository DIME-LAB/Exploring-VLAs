import ast
import sys
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
import json
import numpy as np
from storage import CANONICAL_ROOT, confined, prepare_environment
from protocol import CommandLedger
from success import (contained, initial_arm, settled_containment, SETTLED_TICKS,
                     INTERIOR_FLOOR_M, INTERIOR_RIM_M, INTERIOR_RADIUS_M)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'vla_SO-ARM101/src/so_arm101_control'))
from so_arm101_control.motion_operation import MotionOperation


def method(path, cls, name):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(MotionOperation=MotionOperation, contextmanager=contextmanager, math=math, threading=threading, time=time, np=np, json=json,
                 String=lambda data:SimpleNamespace(data=data))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class SafetyTests(unittest.TestCase):
    def test_command_retry_cannot_advance_physics_twice(self):
        ledger=CommandLedger();writes=[];steps=[]
        payload=dict(epoch=3,observation_stamp_ns=1000000000,command=[0]*6,advance=True)
        def apply(values,stamp):
            writes.append(values)
            return dict(application_stamp_ns=stamp,applied_command=values)
        first=ledger.process(payload,apply,lambda:steps.append(1))
        self.assertEqual(ledger.process(dict(payload),apply,lambda:steps.append(1)),first)
        self.assertEqual((len(writes),len(steps)),(1,1))
        with self.assertRaises(ValueError):
            ledger.process(dict(payload,command=[.01]*6),apply,lambda:None)
        with self.assertRaises(ValueError):
            ledger.process(dict(payload,observation_stamp_ns=0),apply,lambda:None)

    def test_cancelled_epoch_cannot_apply_or_advance(self):
        received=[]
        callback=method(Path(__file__).with_name('collection_runtime.py'),'Runtime','command_received')
        fake=SimpleNamespace(epoch=2,ack_pub=SimpleNamespace(publish=lambda msg:received.append(json.loads(msg.data))))
        payload=dict(epoch=1,observation_stamp_ns=100,command=[0]*6,advance=True)
        callback(fake,SimpleNamespace(data=json.dumps(payload)))
        self.assertIn('Cancelled command epoch',received[0]['error'])

    def test_native_application_rejects_a_five_second_stale_observation(self):
        apply=method(Path(__file__).with_name('collection_runtime.py'),'Runtime','apply')
        fake=SimpleNamespace(busy=False,timeline=SimpleNamespace(is_playing=lambda:False),stamp=lambda:6000000000)
        with self.assertRaisesRegex(RuntimeError,'Stale command observation'):
            apply(fake,[0]*6,1000000000)

    def test_one_frame_callback_rejects_unexpected_extra_physics(self):
        callback=method(Path(__file__).with_name('collection_runtime.py'),'Runtime','after_update')
        pauses=[];published=[]
        fake=SimpleNamespace(mode='collect',busy=True,start_time=0.,world=SimpleNamespace(current_time=2/30),
            timeline=SimpleNamespace(pause=lambda:pauses.append(True)),publish=lambda:published.append(True))
        callback(fake,None)
        self.assertIn('One-frame step advanced',fake.failure)
        self.assertFalse(fake.busy);self.assertFalse(published)

    def test_initial_arm_uses_measured_position_stability_before_velocity_reset(self):
        rows=[dict(state_source='isaac_articulation',state=[0]*6,velocity=[.3]*6) for _ in range(6)]
        self.assertTrue(initial_arm(rows,[0]*5)['passed'])
        rows[-1]['state'][0]=.004
        self.assertFalse(initial_arm(rows,[0]*5)['passed'])

    def snapshot(self, z=.012):
        corners = [[x,y,z] for x in (-.0155,.0155) for y in (-.0103,.0103) for z in (-.0064,.0064)]
        return dict(sim_ns=1000000000, bodies={
            'red_2x3_0':dict(position=[0,0,z],orientation=[1,0,0,0],velocity=[0]*6),
            'cup_red':dict(position=[0,0,0],orientation=[1,0,0,0],velocity=[0]*6)},
            local_corners={'red_2x3_0':corners})

    def test_above_rim_is_rejected(self):
        self.assertFalse(contained(self.snapshot(.104), 'red_2x3_0', 'red')['passed'])

    def test_interior_bounds_are_conservative_against_cup_asset(self):
        import struct
        raw=(ROOT/'vla_SO-ARM101/src/so_arm101_description/meshes/cup/cup.stl').read_bytes()
        count=struct.unpack('<I',raw[80:84])[0]
        data=np.frombuffer(raw,offset=84,count=count,dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')]))
        centers=data['v'].mean(1);normals=data['normal']
        floor=(normals[:,2]>.9)&(np.linalg.norm(centers[:,:2],axis=1)<20)
        asset_floor=float(data['v'][floor,:,:,][...,2].max())/1000
        self.assertLessEqual(INTERIOR_FLOOR_M,asset_floor)
        self.assertTrue(contained(self.snapshot(asset_floor+.0064),'red_2x3_0','red')['passed'])
        self.assertLess(INTERIOR_RIM_M,float(data['v'][...,2].max())/1000)
        inward=((normals[:,:2]*centers[:,:2]).sum(1)<0)&(np.abs(normals[:,2])<.9)&(centers[:,2]>1.19)
        support=-(data['v'][inward,:,:2]*normals[inward,None,:2]).sum(-1)/np.linalg.norm(normals[inward,:2],axis=1)[:,None]
        self.assertLess(INTERIOR_RADIUS_M,float(support.min())/1000)

    def test_floor_supported_block_is_valid_with_authored_rest_offset(self):
        from success import PHYSICAL_FLOOR_M
        for bottom in (PHYSICAL_FLOOR_M, PHYSICAL_FLOOR_M-.0005):
            rows=[self.snapshot(bottom+.0064) for _ in range(SETTLED_TICKS)]
            for index,row in enumerate(rows):row['sim_ns']+=round(index*1e9/30)
            self.assertTrue(settled_containment(rows,'red_2x3_0','red')['passed'])
        self.assertFalse(contained(self.snapshot(INTERIOR_FLOOR_M+.0064-.0001),'red_2x3_0','red')['passed'])
        self.assertFalse(contained(self.snapshot(-.0064),'red_2x3_0','red')['passed'])

    def test_block_corners_not_just_center_must_fit(self):
        sample = self.snapshot(); sample['bodies']['red_2x3_0']['position'][0] = .020
        self.assertFalse(contained(sample, 'red_2x3_0', 'red')['passed'])

    def test_containment_uses_cup_local_frame_and_settling(self):
        rows = [self.snapshot() for _ in range(SETTLED_TICKS)]
        for index, row in enumerate(rows):
            row['sim_ns'] += round(index*1e9/30)
            for body in row['bodies'].values():
                body['position'][0] += .31;body['position'][1] -= .15
                body['orientation'] = [math.sqrt(.5),0,0,math.sqrt(.5)]
        self.assertTrue(settled_containment(rows,'red_2x3_0','red')['passed'])
        rows[-1]['bodies']['red_2x3_0']['velocity'][0] = .1
        self.assertFalse(settled_containment(rows,'red_2x3_0','red')['passed'])

    def test_gui_recoverable_warning_does_not_override_motion_success(self):
        path = ROOT/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        execute = method(path, 'PickPlaceNode', '_qs_execute_step')
        wait = method(path, 'PickPlaceNode', '_qs_wait_for_step')
        node = SimpleNamespace(_qs_abort_evt=threading.Event(), _qs_resume_evt=threading.Event(),
                               _append_log=lambda *args:None)
        node._new_motion_event = threading.Event
        node._begin_motion = lambda event:setattr(node,'_motion_event',event)
        node._current_motion_event = lambda:node._motion_event
        from contextlib import nullcontext
        node._motion_guard = nullcontext
        node._require_motion_generation = lambda:None
        node._set_motion_status = lambda status:setattr(node,'_last_motion_status',status)
        node._motion_verdict = lambda event:getattr(node,'_last_motion_status',None)
        node._qs_resume_evt.set()
        node._qs_wait_for_step = lambda timeout_s:wait(node,timeout_s)
        for success in (True,False):
            def handler():
                node._cmd_error = 'tier1 rejected; fallback evaluated'
                node._last_motion_status = dict(ok=success, outcome='tier2')
                node._motion_event.set()
            node._cmd_probe = handler
            self.assertEqual(execute(node, '_cmd_probe', {}), success)

    def test_held_physical_simulation_needs_no_velocity_wait(self):
        path = ROOT/'vla_SO-ARM101/src/so_arm101_control/so_arm101_control/pick_place.py'
        wait = method(path, 'PickPlaceNode', '_wait_arm_at_rest')
        self.assertTrue(wait(SimpleNamespace(_simulation_held=lambda:True)))

    def test_missing_cup_refuses_without_topology_repair(self):
        runtime = Path(__file__).with_name('collection_runtime.py')
        require = method(runtime, 'Runtime', 'require_scene')
        fake = SimpleNamespace(body_paths={'cup_red':'/red','cup_green':'/green'},
            stage=SimpleNamespace(GetPrimAtPath=lambda path:SimpleNamespace(IsValid=lambda:True)))
        with self.assertRaisesRegex(RuntimeError, 'Fresh scene required'):
            require(fake)

    def test_output_cannot_escape_canonical_root(self):
        self.assertEqual(confined(CANONICAL_ROOT/'datasets/run'), CANONICAL_ROOT/'datasets/run')
        for path in ('/tmp/data', CANONICAL_ROOT/'../../outside'):
            with self.assertRaises(ValueError):confined(path)

    def test_environment_escape_is_refused_before_runtime_imports(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'SOARM_COLLECTION_ROOT':'/tmp/not-collection'}):
            with self.assertRaises(ValueError):prepare_environment(0)

    def test_deploy_client_uses_measured_topic_option(self):
        import importlib.util
        path = ROOT/'linux-env/scripts/run_async_inference_sim.py'
        spec = importlib.util.spec_from_file_location('collection_deploy_probe',path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from unittest.mock import patch
        with patch.object(module.sys, 'argv', [str(path),'--fps','30','--joint-states-topic','/joint_states_measured']):
            args = module.parse_args()
        _, command = module.build_commands(args,python='python')
        self.assertIn('--robot.joint_states_topic=/joint_states_measured',command)
        self.assertIn('--fps=30.0',command)
        self.assertTrue(all(c['fps']==30 and (c['width'],c['height'],c['encoding'])==(640,480,'rgb8')
                            for c in json.loads(args.cameras_json).values()))

    def test_nondefault_gpu_builds_matching_renderer_and_physics_flags(self):
        import subprocess
        source=(ROOT/'linux-env/scripts/isaac/isaacsim_launch.sh').read_text()
        start=source.index('    physics_gpu=0')
        end=source.index('    # shellcheck disable=SC2206',start)
        script='set -eu\nargs=()\nGPU=1\nINSTANCE_INDEX=1\nINSTANCE_ROOT=/mnt/a4500-backup/exploring-vlas/runtime/probe\nheadless=1\nEXT_FOLDER=exts\nKIT_MCP_ROOT=mcp\next=soarm101-dt\nRATE_HZ=0\n'
        script+=source[start:end]+'\nprintf "%s\\n" "${args[@]}"\n'
        result=subprocess.run(['bash','-c',script],check=True,capture_output=True,text=True).stdout.splitlines()
        self.assertIn('--/renderer/activeGpu=1',result)
        self.assertIn('--/physics/cudaDevice=1',result)

    def test_indexed_endpoints_override_inherited_default_domain(self):
        import subprocess
        source=(ROOT/'linux-env/scripts/isaac/isaacsim_launch.sh').read_text().splitlines()
        lines=[next(line for line in source if line.startswith('    export '+key+'='))
               for key in ('ROS_DOMAIN_ID','OMNI_KIT_MCP_PORT')]
        script='INSTANCE_INDEX=1\nROS_DOMAIN_ID=0\nOMNI_KIT_MCP_PORT=8767\n'+'\n'.join(lines)+'\nprintf "%s %s" "$ROS_DOMAIN_ID" "$OMNI_KIT_MCP_PORT"'
        result=subprocess.run(['bash','-c',script],capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout,'91 8779')


if __name__ == '__main__':unittest.main()
