"""Real bounded transfer/dirfd entrances with local CPU-only SSH transports."""
import contextlib,hashlib,io,json,os,shlex,subprocess,sys,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import backup_space as space
import triplet_production as prod


class BackupSpaceTests(unittest.TestCase):
    def fixture(self,root):
        base=root/'owned';projection=base/'datasets/counterfactual-production-v1';projection.mkdir(parents=True)
        source=root/'source';source.mkdir();(source/'blob').write_bytes(b'abc')
        return source,base,projection,projection/'new-group'

    @contextlib.contextmanager
    def local_transport(self,root,base,projection,free=500*1024**3,after_probe=None,before_transfer=None,receiver_open=None):
        events=[];receipts=[];native_open=os.open
        def execute(code,wire=None):
            output=io.StringIO()
            measured=free() if callable(free) else free
            with patch.object(os,'fstatvfs',return_value=NS(f_bavail=measured,f_frsize=1)),contextlib.redirect_stdout(output):
                if wire is None:exec(code,{})
                else:
                    with patch.object(sys,'stdin',NS(buffer=wire)),patch.object(os,'open',side_effect=receiver_open or native_open):exec(code,{})
            return output.getvalue()
        def probe(argv,**kw):
            events.append('probe');result=execute(shlex.split(argv[-1])[2])
            if after_probe:result=after_probe(result) or result
            return result
        def transfer(argv,**kw):
            events.append('transfer')
            if before_transfer:before_transfer()
            try:result=execute(shlex.split(argv[-1])[2],kw['stdin'])
            except Exception as exc:raise subprocess.CalledProcessError(1,argv,stderr=repr(exc)) from exc
            receipts.append(json.loads(result));return NS(stdout=result,returncode=0)
        with patch.object(prod,'ROOT',root),patch.object(prod,'DUAL',projection),\
             patch.object(space,'BASE',base),patch.object(space,'PROJECTION',projection),\
             patch.object(prod.subprocess,'check_output',side_effect=probe),patch.object(prod.subprocess,'run',side_effect=transfer):
            yield events,receipts

    def test_full_incoming_size_and_metadata_must_fit_above_100_gib(self):
        size=100*1024**2;count=68;minimum=100*1024**3+size+64*1024**2+count*16384
        result=space.require_capacity(minimum,size,count)
        self.assertEqual(result['projected_free_bytes'],100*1024**3)
        with self.assertRaisesRegex(RuntimeError,'100 GiB'):space.require_capacity(minimum-1,size,count)
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):space.require_capacity(bad,size,count)

    def test_actual_remote_probe_is_read_only_and_refuses_low_space(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root)
            with self.local_transport(root,base,projection) as (events,receipts):
                out=io.StringIO()
                with patch.object(os,'fstatvfs',return_value=NS(f_bavail=500*1024**3,f_frsize=1)),contextlib.redirect_stdout(out):exec(space.probe_code(dest,123,1),{})
                result=json.loads(out.getvalue());self.assertEqual(result['incoming_bytes'],123)
                self.assertEqual(result['directory_identities'][-1]['path'],str(projection))
                self.assertFalse(dest.exists())
                with patch.object(os,'fstatvfs',return_value=NS(f_bavail=100*1024**3,f_frsize=1)):
                    with self.assertRaisesRegex(RuntimeError,'100 GiB'):exec(space.probe_code(dest,123,1),{})
                self.assertFalse(dest.exists())

    def test_static_symlink_refuses_without_following_its_target(self):
        for dangling in (False,True):
            with self.subTest(dangling=dangling),tempfile.TemporaryDirectory() as d:
                root=Path(d);source,base,projection,dest=self.fixture(root);outside=root/'outside'
                if not dangling:outside.mkdir()
                dest.symlink_to(outside,target_is_directory=True)
                with self.local_transport(root,base,projection),patch.object(os,'fstatvfs',side_effect=AssertionError('No space read through a link')):
                    with self.assertRaisesRegex(ValueError,'real directory'):exec(space.probe_code(dest/'child',0,0),{})
                self.assertEqual(list(outside.iterdir()) if outside.exists() else [],[])

    def test_our_copies_serialize_and_symlink_lock_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'copy.lock';waiting=threading.Event();entered=threading.Event()
            def second():
                waiting.set()
                with space.serialized_copy(path):entered.set()
            with space.serialized_copy(path):
                worker=threading.Thread(target=second);worker.start();self.assertTrue(waiting.wait(2));self.assertFalse(entered.wait(.05))
            worker.join(2);self.assertFalse(worker.is_alive());self.assertTrue(entered.is_set())
            link=Path(d)/'link';link.symlink_to(path)
            with self.assertRaises(OSError):
                with space.serialized_copy(link):self.fail('Link admitted')

    def test_real_backup_entry_refuses_low_capacity_before_destination_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source)
            with self.local_transport(root,base,projection,free=100*1024**3) as (events,receipts):
                with self.assertRaisesRegex(RuntimeError,'100 GiB'):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(events,['probe']);self.assertFalse(dest.exists());self.assertEqual(receipts,[])

    def test_real_bounded_backup_receipt_binds_all_files_capacity_and_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root)
            (source/'nested').mkdir();(source/'nested/zero').write_bytes(b'');(source/'nested/binary').write_bytes(bytes(range(256))*5000)
            files=prod.manifest(source)
            with self.local_transport(root,base,projection) as (events,receipts):
                result=prod.backup(source,files,dest,io.StringIO())
                # Existing regular files may be recopied with the same admitted bytes.
                repeat=prod.backup(source,files,dest,io.StringIO())
            self.assertTrue(result['passed']);self.assertTrue(repeat['passed']);self.assertEqual(events,['probe','transfer','probe','transfer'])
            self.assertEqual(prod.manifest(dest),files);self.assertEqual(result['transferred_bytes'],sum((source/n).stat().st_size for n in files))
            self.assertEqual(result['capacity_admission']['incoming_bytes'],result['transferred_bytes'])
            self.assertEqual(result['destination_identity']['inode'],dest.stat().st_ino)
            self.assertEqual(result['files_sha256'],files)

    def test_review_unlisted_file_before_backup_refuses_before_probe_or_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source)
            (source/'late-file').write_bytes(b'x'*(2*1024**2))
            with self.local_transport(root,base,projection,free=100*1024**3+64*1024**2+16384+4) as (events,receipts):
                with self.assertRaisesRegex(ValueError,'Source changed before'):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(events,[]);self.assertFalse(dest.exists())

    def test_late_file_or_growth_after_payload_seal_never_transfers_unadmitted_bytes(self):
        for variant in ('addition','growth'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source)
                def change():
                    with (source/('late' if variant=='addition' else 'blob')).open('ab') as f:f.write(b'x'*(2*1024**2))
                with self.local_transport(root,base,projection,free=100*1024**3+64*1024**2+16384+4,before_transfer=change) as (events,receipts):
                    with self.assertRaisesRegex(ValueError,'Source changed during'):prod.backup(source,files,dest,io.StringIO())
                self.assertEqual(events,['probe','transfer']);self.assertEqual(receipts[0]['transferred_bytes'],3)
                self.assertEqual((dest/'blob').read_bytes(),b'abc');self.assertFalse((dest/'late').exists())
                self.assertEqual(receipts[0]['capacity_admission']['incoming_bytes'],3)

    def test_source_growth_after_probe_refuses_before_destination_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source)
            def change(reply):(source/'blob').write_bytes(b'longer')
            with self.local_transport(root,base,projection,after_probe=change) as (events,receipts):
                with self.assertRaisesRegex(ValueError,'source size/type changed'):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(events,['probe']);self.assertFalse(dest.exists())

    def test_review_destination_symlink_after_probe_refuses_and_outside_stays_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source);outside=root/'outside';outside.mkdir()
            def swap(reply):dest.symlink_to(outside,target_is_directory=True)
            with self.local_transport(root,base,projection,after_probe=swap) as (events,receipts):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(events,['probe','transfer']);self.assertEqual(list(outside.iterdir()),[]);self.assertEqual(receipts,[])

    def test_destination_swap_after_open_is_detected_on_pinned_descriptor_chain(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source);outside=root/'outside';outside.mkdir();native=os.open;swapped=[]
            def swap(path,flags,*a,**kw):
                fd=native(path,flags,*a,**kw)
                if path==dest.name and flags&os.O_DIRECTORY and not swapped:
                    swapped.append(True);dest.rename(projection/'retained-original');dest.symlink_to(outside,target_is_directory=True)
                return fd
            with self.local_transport(root,base,projection,receiver_open=swap) as (events,receipts):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,dest,io.StringIO())
            self.assertTrue(swapped);self.assertEqual(list(outside.iterdir()),[]);self.assertEqual(receipts,[])

    def test_regular_directory_replacement_after_probe_also_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source);dest.mkdir()
            def swap(reply):dest.rename(projection/'retained-original');dest.mkdir()
            with self.local_transport(root,base,projection,after_probe=swap) as (events,receipts):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(list(dest.iterdir()),[]);self.assertEqual(receipts,[])

    def test_wrong_copy_or_floor_receipt_refuses_before_transfer(self):
        for field,value in (('incoming_bytes',0),('floor_bytes',0),('destination','/foreign'),('filesystem_root','/foreign'),('directory_identities',[])):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source)
                def change(reply):
                    row=json.loads(reply);row[field]=value;return json.dumps(row)
                with self.local_transport(root,base,projection,after_probe=change) as (events,receipts):
                    with self.assertRaises(RuntimeError):prod.backup(source,files,dest,io.StringIO())
                self.assertEqual(events,['probe']);self.assertFalse(dest.exists())


    def test_capacity_drop_after_probe_refuses_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root);files=prod.manifest(source);free=[500*1024**3]
            def drop(reply):free[0]=100*1024**3
            with self.local_transport(root,base,projection,free=lambda:free[0],after_probe=drop) as (events,receipts):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(events,['probe','transfer']);self.assertFalse(dest.exists());self.assertEqual(receipts,[])

    def test_nested_destination_link_refuses_without_following_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root)
            (source/'nested').mkdir();(source/'nested/blob').write_bytes(b'nested')
            dest.mkdir();outside=root/'outside';outside.mkdir();(dest/'nested').symlink_to(outside,target_is_directory=True)
            files=prod.manifest(source)
            with self.local_transport(root,base,projection) as (events,receipts):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup(source,files,dest,io.StringIO())
            self.assertEqual(list(outside.iterdir()),[]);self.assertEqual(receipts,[])

    def test_real_child_receiver_reads_binary_framing_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,base,projection,dest=self.fixture(root)
            (source/'video').write_bytes(bytes(range(256))*9000);files=prod.manifest(source);native=subprocess.run
            prefix='import os,types;os.fstatvfs=lambda fd:types.SimpleNamespace(f_bavail=500*1024**3,f_frsize=1)\n'
            def probe(argv,**kw):
                return native([sys.executable,'-B','-c',prefix+shlex.split(argv[-1])[2]],capture_output=True,text=True,check=True).stdout
            def receiver(argv,**kw):
                return native([sys.executable,'-B','-c',prefix+shlex.split(argv[-1])[2]],**kw)
            with patch.object(prod,'ROOT',root),patch.object(prod,'DUAL',projection),\
                 patch.object(space,'BASE',base),patch.object(space,'PROJECTION',projection),\
                 patch.object(prod.subprocess,'check_output',side_effect=probe),patch.object(prod.subprocess,'run',side_effect=receiver),\
                 (root/'backup.log').open('w') as log:
                receipt=prod.backup(source,files,dest,log)
            self.assertEqual(prod.manifest(dest),files);self.assertEqual(receipt['transferred_bytes'],2304003)
            self.assertTrue(receipt['passed'])


    def registry_fixture(self,root):
        source,base,projection,dest=self.fixture(root)
        production=root/'datasets/counterfactual-targets/production-v1';production.mkdir(parents=True)
        runtime=root/'runtime/session';runtime.mkdir(parents=True)
        for split,seed in (('train',720000),('heldout',730000)):
            raw=root/f'datasets/counterfactual-targets/{split}/layout-{seed}'
            value=dict(cpu_fixture_only=True,role='authoritative-rtx-native',sealed=True,production_root=str(production),finalization_path=str(production/'production-finalization.json'),
                wave_ledger_path=str(production/'wave-ledger.json'),layouts=[dict(requested_layout_seed=seed,dataset_root=str(raw/'dataset'),raw_root=str(raw),validation_path=str(raw/'validation.json'))],failed_groups=[])
            prod.atomic_json(production/(split+'-registry.json'),value)
        return base,projection,dict(production_root=str(production),runtime_root=str(runtime))

    def test_derived_registry_writes_use_the_same_pinned_bounded_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,projection,recipe=self.registry_fixture(root)
            with self.local_transport(root,base,projection) as (events,receipts),patch.object(prod,'publication_gate'):
                receipt=prod.backup_registries(recipe)
            self.assertEqual(events,['probe','transfer','probe','transfer'])
            self.assertTrue(receipt['projection_backup']['passed'])
            for split in ('train','heldout'):
                info=receipt['projections'][split];path=Path(info['path'])
                self.assertEqual(prod.sha(path),info['sha256'])
                self.assertEqual(json.loads(path.read_text())['authoritative_registry_sha256'],info['authoritative_registry_sha256'])

    def test_derived_registry_destination_swap_cannot_bypass_pinned_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,projection,recipe=self.registry_fixture(root);outside=root/'outside';outside.mkdir();calls=[]
            def swap(reply):
                calls.append(True)
                if len(calls)==2:
                    destination=Path(json.loads(reply)['destination'])
                    destination.rename(destination.with_name('retained-original'));destination.symlink_to(outside,target_is_directory=True)
            with self.local_transport(root,base,projection,after_probe=swap),patch.object(prod,'publication_gate'):
                with self.assertRaises(subprocess.CalledProcessError):prod.backup_registries(recipe)
            self.assertEqual(len(calls),2);self.assertEqual(list(outside.iterdir()),[])
