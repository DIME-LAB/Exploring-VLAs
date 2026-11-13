import ast
import copy
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from storage import CANONICAL_ROOT, confined
from success import contained, settled_containment, SETTLED_TICKS

ROOT = Path(__file__).resolve().parents[3]


def method(path, cls, name):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(threading=threading, time=time)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class SafetyTests(unittest.TestCase):
    def snapshot(self, z=.012):
        corners = [[x,y,z] for x in (-.0155,.0155) for y in (-.0103,.0103) for z in (-.0064,.0064)]
        return dict(sim_ns=1000000000, bodies={
            'red_2x3_0':dict(position=[0,0,z],orientation=[1,0,0,0],velocity=[0]*6),
            'cup_red':dict(position=[0,0,0],orientation=[1,0,0,0],velocity=[0]*6)},
            local_corners={'red_2x3_0':corners})

    def test_above_rim_is_rejected(self):
        self.assertFalse(contained(self.snapshot(.104), 'red_2x3_0', 'red')['passed'])

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
        node._qs_resume_evt.set()
        node._qs_wait_for_step = lambda timeout_s:wait(node,timeout_s)
        for success in (True,False):
            def handler():
                node._cmd_error = 'tier1 rejected; fallback evaluated'
                node._last_motion_status = dict(ok=success, outcome='tier2')
                node._motion_event.set()
            node._cmd_probe = handler
            self.assertEqual(execute(node, '_cmd_probe', {}), success)

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

    def test_deploy_client_uses_measured_topic_option(self):
        import importlib.util
        path = ROOT/'linux-env/scripts/run_async_inference_sim.py'
        spec = importlib.util.spec_from_file_location('collection_deploy_probe',path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        args = module.build_parser().parse_args(['--fps','30','--joint-states-topic','/joint_states_measured'])
        _, command = module.build_commands(args,python='python')
        self.assertIn('--robot.joint_states_topic=/joint_states_measured',command)
        self.assertIn('--fps=30.0',command)

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


if __name__ == '__main__':unittest.main()
