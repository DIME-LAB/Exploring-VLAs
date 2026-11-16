"""Replay and export agree with the immutable mode declared in run.json."""
import argparse,ast,io,json,math,os,tempfile,unittest
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from checkpoint import require_episode_mode,require_recovery_mode
from contract import parameters
from target_triplets import configure_args as configure_triplet_args,validate_run as validate_triplet_run
from scene_bank import configure_args as configure_scene_bank_args


class ReplayModeTests(unittest.TestCase):
    def test_replay_refuses_mode_mismatch_before_any_ros_or_instance_write(self):
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        cut=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='init')
        with tempfile.TemporaryDirectory() as directory:
            receipt=Path(directory)/'receipt.json';receipt.write_text(json.dumps(dict(params=parameters(1,'red',regrasp=True))))
            argv=['collect.py','--instance','0','--output',directory,'--normal-only','--replay-receipt',str(receipt)]
            scope=dict(argparse=argparse,__doc__='mode',Path=Path,math=math,json=json,require_episode_mode=require_episode_mode,
                configure_triplet_args=configure_triplet_args,configure_scene_bank_args=configure_scene_bank_args)
            with patch('sys.argv',argv),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit) as error:
                exec(compile(ast.Module(body=main.body[:cut],type_ignores=[]),'actual replay mode entrance','exec'),scope)
            self.assertEqual(error.exception.code,2)

    def test_actual_export_refuses_inconsistent_episode_before_environment_or_dataset_creation(self):
        tree=ast.parse(Path(__file__).with_name('export.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        cut=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='prepare_environment')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'run.json').write_text(json.dumps(dict(normal_only=True,regrasp_only=False,recovery_mode='normal_only',instance=0)))
            (root/'summary.json').write_text(json.dumps(dict(complete=True,end_marker='COLLECTION_FINISHED')))
            attempt=root/'attempt-0000';attempt.mkdir();(attempt/'receipt.json').write_text(json.dumps(dict(accepted=True,params=parameters(1,'red',regrasp=True))))
            scope=dict(argparse=argparse,__doc__='export',Path=Path,json=json,time=time,atomic_json=lambda p,v:p.write_text(json.dumps(v)),validate_triplet_run=validate_triplet_run,
                confined=lambda p:p,require_finalized=lambda p:None,require_recovery_mode=require_recovery_mode,require_episode_mode=require_episode_mode)
            with patch('sys.argv',['export.py',str(root),'--report-dir',str(root/'report')]),self.assertRaisesRegex(ValueError,'declared collection mode'):
                exec(compile(ast.Module(body=main.body[:cut],type_ignores=[]),'actual export mode entrance','exec'),scope)
            self.assertFalse((root/'dataset').exists())

    def test_each_declared_mode_accepts_only_its_episode_kinds(self):
        for only_normal,only_regrasp in ((False,False),(True,False),(False,True)):
            for regrasp in (False,True):
                params=parameters(1,'red',regrasp=regrasp);invalid=(only_normal and regrasp) or (only_regrasp and not regrasp)
                if invalid:
                    with self.assertRaises(ValueError):require_episode_mode(params,only_regrasp,only_normal)
                else:require_episode_mode(params,only_regrasp,only_normal)
