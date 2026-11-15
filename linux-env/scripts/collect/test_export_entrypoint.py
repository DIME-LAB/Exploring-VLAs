"""A private frozen checkout exports only through its own pinned entrypoint."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from storage import CANONICAL_ROOT


class ExportEntrypointTests(unittest.TestCase):
    def load(self):
        path=Path(__file__).with_name('hosts')/'rtx6000_export.py'
        spec=importlib.util.spec_from_file_location('qualified_export_wrapper',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def invoke(self,wrapper,target,report):
        m=self.load();pin=report.parent/'pin.json';pin.write_text('[]')
        with patch.object(m,'__file__',str(wrapper)),patch.object(m,'ROOT',CANONICAL_ROOT),patch.object(m,'PIN',pin),\
                patch.object(m,'require_fork',return_value=m.CUT),patch.object(m,'snapshot',return_value=[]),\
                patch.object(sys,'argv',['rtx6000_export.py',str(target),'--report-dir',str(report)]),\
                patch.object(sys,'path',list(sys.path)),patch.object(m.runpy,'run_path') as execute:
            m.main()
            return execute.call_args
    def test_exact_failed_private_checkout_now_accepts_own_export(self):
        wrapper=Path('/mnt/storage3/aaugus11/exploring-vlas/repo/regrasp-production-1726fe9/linux-env/scripts/collect/hosts/rtx6000_export.py')
        target=wrapper.parents[1]/'export.py'
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='export-entry-') as d:
            report=Path(d)/'report';call=self.invoke(wrapper,target,report)
            self.assertEqual(call.args[0],str(target));self.assertEqual(call.kwargs,dict(run_name='__main__'))
            self.assertTrue(json.loads((report/'export-environment.json').read_text())['passed'])
    def test_other_checkout_other_script_and_linked_foreign_leaf_refuse(self):
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='export-entry-') as d:
            d=Path(d);wrapper=d/'own/linux-env/scripts/collect/hosts/rtx6000_export.py';wrapper.parent.mkdir(parents=True);wrapper.touch()
            own=wrapper.parents[1]/'export.py';own.touch();foreign=d/'other/export.py';foreign.parent.mkdir();foreign.touch()
            for target in (foreign,own.with_name('collect.py')):
                with self.subTest(target=target),self.assertRaisesRegex(RuntimeError,'Unexpected collector export entrypoint'):
                    self.invoke(wrapper,target,d/'report')
            own.unlink();own.symlink_to(foreign)
            with self.assertRaisesRegex(RuntimeError,'Unexpected collector export entrypoint'):self.invoke(wrapper,own,d/'report')
    def test_export_runner_preserves_original_traceback_without_a_receipt(self):
        from regrasp_runner import Processes
        from run_smoke import StartupFailure
        with tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'tmp',prefix='export-entry-') as d:
            p=object.__new__(Processes);p.config=dict(instance=3);p.wrapper='not-executed';p.root=lambda c:Path(d)
            log=Path(d)/'export.log';log.write_text('RuntimeError: Unexpected collector export entrypoint: original-path\n')
            p.stage=lambda *a:dict(returncode=1,log=str(log))
            with self.assertRaisesRegex(StartupFailure,'Unexpected collector export entrypoint: original-path'):
                p.export(dict(index=0))


if __name__=='__main__':unittest.main()
