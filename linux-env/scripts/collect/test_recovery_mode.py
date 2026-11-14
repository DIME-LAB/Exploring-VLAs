"""Normal-only runs preserve balance and cannot resume under a different mode."""
import argparse,ast,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from checkpoint import progress,episode_regrasp,require_recovery_mode
from contract import COLORS,parameters


class RecoveryModeTests(unittest.TestCase):
    def test_fifty_normal_accepts_remain_normal_and_balanced_across_resume(self):
        receipts=[]
        for trial in range(50):
            self.assertFalse(episode_regrasp(trial,normal_only=True))
            receipts.append(dict(params=dict(parameters(350000+trial,COLORS[trial%3],regrasp=False),trial=trial),accepted=True,timing=dict(passed=True)))
            self.assertEqual(progress(receipts,350000,normal_only=True),(trial+1,trial+1))
        self.assertEqual([sum(r['params']['color']==c for r in receipts) for c in COLORS],[17,17,16])
        with self.assertRaisesRegex(ValueError,'parameters differ'):progress(receipts,350000)

    def test_default_and_dedicated_recovery_behavior_unchanged(self):
        self.assertFalse(episode_regrasp(4));self.assertTrue(episode_regrasp(5))
        self.assertEqual(sum(episode_regrasp(i) for i in range(100)),18)
        self.assertTrue(all(episode_regrasp(i,regrasp_only=True) for i in range(50)))

    def test_resume_modes_are_distinct_even_before_first_recovery_slot(self):
        for old in ('mixed','normal_only','regrasp_only'):
            prior=dict(regrasp_only=old=='regrasp_only',normal_only=old=='normal_only',recovery_mode=old)
            for new in ('mixed','normal_only','regrasp_only'):
                if old==new:self.assertEqual(require_recovery_mode(prior,new=='regrasp_only',new=='normal_only'),new)
                else:
                    with self.assertRaisesRegex(ValueError,'mode differs'):require_recovery_mode(prior,new=='regrasp_only',new=='normal_only')
        self.assertEqual(require_recovery_mode({'regrasp_only':False}),'mixed')
        with self.assertRaisesRegex(ValueError,'inconsistent'):require_recovery_mode(dict(recovery_mode='normal_only'))

    def test_actual_run_refuses_mode_mismatch_before_begin_or_mutation(self):
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Collector');run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        stop=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='mkdir')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prior=root/'run.json';raw=json.dumps(dict(regrasp_only=False,normal_only=True,recovery_mode='normal_only'));prior.write_text(raw)
            events=[];scope=dict(self=NS(args=NS(output=root,resume=True,regrasp_only=False,normal_only=False)),confined=lambda p:p,json=json,collection_status=NS(require_finalized=lambda p:events.append('validated'),begin=lambda p:self.fail('begin changed predecessor')),require_recovery_mode=require_recovery_mode)
            with self.assertRaisesRegex(ValueError,'mode differs'):exec(compile(ast.Module(body=run.body[:stop],type_ignores=[]),'actual run mode entrance','exec'),scope)
            self.assertEqual(prior.read_text(),raw);self.assertEqual(events,['validated'])

    def test_actual_parser_refuses_conflicting_flags(self):
        tree=ast.parse(Path(__file__).with_name('collect.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        stop=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets))
        scope=dict(argparse=argparse,__doc__='mode test',Path=Path)
        exec(compile(ast.Module(body=main.body[:stop],type_ignores=[]),'actual CLI parser','exec'),scope)
        parser=scope['parser'];base=['--instance','0','--output','/unused']
        self.assertTrue(parser.parse_args(base+['--normal-only']).normal_only)
        with self.assertRaises(SystemExit):parser.parse_args(base+['--normal-only','--regrasp-only'])
