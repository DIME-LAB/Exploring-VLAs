"""Exercise recording provenance through the real collector comparison."""
import hashlib
from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace as NS
import unittest
import finger_geometry,miss_clearance
from cup_geometry import PROFILE
from source_provenance import loaded_sources,recording_manifest,source_digest,verify_sources
from test_review_regressions import load_class

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


class SourceProvenanceTests(unittest.TestCase):
    def collector(self):
        Collector,scope=load_class('collect.py','Collector')
        scope.update(REPO=REPO,PROFILE=PROFILE,recording_manifest=recording_manifest,_COLLECT_LOADED_SOURCES={})
        node=object.__new__(Collector)
        node.executor=NS(image_take=NS(metadata=dict(source_sha256=source_digest(HERE/'image_take.c'),
            python_source_sha256=source_digest(HERE/'image_transport.py'))))
        native=dict(loaded_sha256=source_digest(HERE/'collection_runtime.py'),
            loaded_dependencies={n:source_digest(HERE/n) for n in ('finger_geometry.py','miss_clearance.py','source_provenance.py')})
        return node,native

    def test_actual_collector_comparison_covers_native_only_finger_and_miss_sources(self):
        node,native=self.collector();sources=node.recording_sources(native)
        for name,digest in native['loaded_dependencies'].items():
            self.assertEqual(sources['linux-env/scripts/collect/'+name],digest)
        for name,digest in loaded_sources(HERE).items():
            self.assertEqual(sources['linux-env/scripts/collect/'+name],digest)

    def test_actual_collector_comparison_refuses_changed_native_sources(self):
        node,native=self.collector();native['loaded_dependencies']['finger_geometry.py']='0'*64
        with self.assertRaisesRegex(RuntimeError,'finger_geometry.py'):node.recording_sources(native)

    def test_actual_collector_comparison_refuses_changed_recorder_source(self):
        node,native=self.collector()
        node.recording_sources.__func__.__globals__['_COLLECT_LOADED_SOURCES']={'recovery_search.py':'0'*64}
        with self.assertRaisesRegex(RuntimeError,'changed.*recovery_search.py'):node.recording_sources(native)

    def test_actual_collector_comparison_refuses_missing_native_sources(self):
        node,native=self.collector();native['loaded_dependencies']['absent_native_source.py']='0'*64
        with self.assertRaisesRegex(RuntimeError,'unavailable.*absent_native_source.py'):node.recording_sources(native)

    def test_loaded_sources_keep_first_fingerprints_and_refuse_changes_or_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'helper.py';p.write_text('value=1\n');modules={'helper':NS(__file__=str(p))}
            first=loaded_sources(directory,modules=modules)
            self.assertEqual(first,{'helper.py':hashlib.sha256(p.read_bytes()).hexdigest()})
            p.write_text('value=2\n')
            with self.assertRaisesRegex(RuntimeError,'changed.*helper.py'):loaded_sources(directory,first,modules)
            p.unlink()
            with self.assertRaisesRegex(RuntimeError,'unavailable.*helper.py'):loaded_sources(directory,first,{})

    def test_native_only_file_is_included_even_when_not_imported_by_recorder(self):
        with tempfile.TemporaryDirectory() as repo:
            directory=Path(repo)/'linux-env/scripts/collect';directory.mkdir(parents=True)
            p=directory/'native_only.py';p.write_text('native=1\n')
            result=recording_manifest(repo,[],{'native_only.py':source_digest(p)},modules={})
            self.assertEqual(result,{'linux-env/scripts/collect/native_only.py':source_digest(p)})

    def test_invalid_or_foreign_local_sources_refuse(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as foreign:
            here=Path(directory)/'helper.py';here.write_text('value=1\n')
            other=Path(foreign)/'helper.py';other.write_text('value=2\n')
            with self.assertRaisesRegex(RuntimeError,'another directory'):
                loaded_sources(directory,modules={'helper':NS(__file__=str(other))})
            for name in ('../helper.py','/helper.py','helper.so'):
                with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'Invalid local'):
                    verify_sources(directory,{name:'0'*64})
            here.unlink();here.symlink_to(other)
            with self.assertRaisesRegex(RuntimeError,'leaves its directory'):
                loaded_sources(directory,modules={'helper':NS(__file__=str(here))})

    def test_unrelated_lazy_modules_are_not_queried_for_synthetic_file_attributes(self):
        module=ModuleType('lazy_namespace')
        def dynamic(name):raise AssertionError('Unrelated lazy attribute was invoked: '+name)
        module.__getattr__=dynamic
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(loaded_sources(directory,modules={'lazy_namespace':module}),{})


if __name__=='__main__':unittest.main()
