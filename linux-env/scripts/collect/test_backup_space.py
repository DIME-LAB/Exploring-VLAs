"""Next-run backup admission; no network, GPU or estate-archive access."""
import contextlib,hashlib,io,json,os,subprocess,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import backup_space as space
import triplet_production as prod


class BackupSpaceTests(unittest.TestCase):
    def test_full_incoming_size_and_metadata_must_fit_above_100_gib(self):
        size=100*1024**2;count=68;minimum=100*1024**3+size+64*1024**2+count*16384
        result=space.require_capacity(minimum,size,count)
        self.assertEqual(result['projected_free_bytes'],100*1024**3)
        with self.assertRaisesRegex(RuntimeError,'100 GiB'):space.require_capacity(minimum-1,size,count)
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):space.require_capacity(bad,size,count)

    def fixture(self,root):
        base=root/'owned';projection=base/'datasets/counterfactual-production-v1';projection.mkdir(parents=True)
        return base,projection,projection/'new-group'
    def run_probe(self,base,projection,destination,free):
        output=io.StringIO()
        with patch.object(space,'BASE',base),patch.object(space,'PROJECTION',projection),\
                patch('os.statvfs',return_value=NS(f_bavail=free,f_frsize=1)),contextlib.redirect_stdout(output):
            exec(space.probe_code(destination,123,1),{})
        return json.loads(output.getvalue())

    def test_actual_remote_probe_is_read_only_and_refuses_low_space(self):
        with tempfile.TemporaryDirectory() as d:
            base,projection,destination=self.fixture(Path(d))
            result=self.run_probe(base,projection,destination,500*1024**3)
            self.assertTrue(result['passed']);self.assertEqual(result['incoming_bytes'],123)
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(RuntimeError,'100 GiB'):self.run_probe(base,projection,destination,100*1024**3)
            self.assertFalse(destination.exists())

    def test_symlink_ancestors_refuse_before_traversing_any_archive_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,projection,_=self.fixture(root);archive=root/'untouched-test-archive';archive.mkdir()
            link=projection/'link';link.symlink_to(archive,target_is_directory=True)
            with patch.object(space,'BASE',base),patch.object(space,'PROJECTION',projection),\
                    patch('os.statvfs',side_effect=AssertionError('Must not probe after encountering link')):
                with self.assertRaisesRegex(ValueError,'real directory'):exec(space.probe_code(link/'child',0,0),{})
            self.assertEqual(list(archive.iterdir()),[])

    def test_our_copies_serialize_and_symlink_lock_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'copy.lock';waiting=threading.Event();entered=threading.Event()
            def second():
                waiting.set()
                with space.serialized_copy(path):entered.set()
            with space.serialized_copy(path):
                worker=threading.Thread(target=second);worker.start();self.assertTrue(waiting.wait(2))
                self.assertFalse(entered.wait(.05))
            worker.join(2);self.assertFalse(worker.is_alive());self.assertTrue(entered.is_set())
            link=Path(d)/'link';link.symlink_to(path)
            with self.assertRaises(OSError):
                with space.serialized_copy(link):self.fail('Link was admitted')

    def test_real_backup_entry_refuses_before_any_destination_mkdir_or_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'blob').write_bytes(b'abc')
            files={'blob':hashlib.sha256(b'abc').hexdigest()}
            with patch.object(prod,'ROOT',root),patch.object(prod.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,'capacity probe')),\
                    patch.object(prod.subprocess,'run') as write:
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,prod.DUAL/'unit-unused',io.StringIO())
                write.assert_not_called()
            with space.serialized_copy(root/'runtime/counterfactual-backup-space.lock'):pass

    def test_real_backup_entry_runs_capacity_before_copy_and_records_its_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'blob').write_bytes(b'abc')
            files={'blob':hashlib.sha256(b'abc').hexdigest()};events=[]
            admission=dict(space.require_capacity(500*1024**3,3,1),destination=str(prod.DUAL/'unit-unused'),filesystem_root=str(space.BASE))
            def probe(*a,**kw):events.append('capacity');return json.dumps(admission)
            def run(argv,**kw):events.append('rsync' if argv[0]=='rsync' else 'remote write/verify')
            with patch.object(prod,'ROOT',root),patch.object(prod.subprocess,'check_output',side_effect=probe),\
                    patch.object(prod.subprocess,'run',side_effect=run):
                result=prod.backup(source,files,prod.DUAL/'unit-unused',io.StringIO())
            self.assertEqual(events[0],'capacity');self.assertEqual(len(events),4)
            self.assertTrue(result['passed']);self.assertEqual(result['capacity_admission'],admission)

    def test_wrong_copy_or_floor_receipt_refuses_before_writes(self):
        for field,value in (('incoming_bytes',0),('floor_bytes',0),('destination','/foreign'),('filesystem_root','/foreign')):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d);source=root/'source';source.mkdir();(source/'blob').write_bytes(b'abc')
                files={'blob':hashlib.sha256(b'abc').hexdigest()}
                admission=dict(space.require_capacity(500*1024**3,3,1),destination=str(prod.DUAL/'unit-unused'),filesystem_root=str(space.BASE))
                admission[field]=value
                with patch.object(prod,'ROOT',root),patch.object(prod.subprocess,'check_output',return_value=json.dumps(admission)),\
                        patch.object(prod.subprocess,'run') as write:
                    with self.assertRaisesRegex(RuntimeError,'not bound'):prod.backup(source,files,prod.DUAL/'unit-unused',io.StringIO())
                    write.assert_not_called()
