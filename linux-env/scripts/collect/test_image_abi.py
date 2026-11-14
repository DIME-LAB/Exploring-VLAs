"""ABI drift and an altered cached binary must refuse before taking a packet."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import image_abi,image_transport

class AbiTests(unittest.TestCase):
    def test_manifest_covers_transitive_headers_native_rmw_and_executor(self):
        expected=json.loads(Path(image_abi.__file__).with_name('image_transport_abi.json').read_text())
        self.assertEqual(image_abi.verify(expected,expected),expected['fingerprint'])
        names=[Path(p['path']).name for p in expected['files']]
        for name in ('serialized_message.h','allocator.h','executors.py','librcl.so','librmw.so','librmw_cyclonedds_cpp.so'):
            self.assertIn(name,names)
        self.assertTrue(any(n.startswith('libddsc.so') for n in names))
        self.assertTrue(any('_rclpy_pybind11' in n for n in names))
        for entry in expected['files']:
            observed=copy.deepcopy(expected)
            next(p for p in observed['files'] if p['path']==entry['path'])['sha256']='0'*64
            observed['fingerprint']=image_abi.fingerprint(observed['files'],observed['compiler'])
            with self.subTest(path=entry['path']),self.assertRaisesRegex(RuntimeError,'Unverified native image ABI'):
                image_abi.verify(observed,expected)

    def test_cached_binary_and_receipt_are_both_verified(self):
        with tempfile.TemporaryDirectory() as d:
            lib=Path(d)/'image_take.so';receipt=Path(d)/'build.json';lib.write_bytes(b'known helper')
            expected=dict(abi_fingerprint='abc',source_sha256='c-source',python_source_sha256='py-source')
            with self.assertRaisesRegex(RuntimeError,'complete build receipt'):image_abi.verify_cache(lib,receipt,expected)
            row=dict(expected,returncode=0,library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest());receipt.write_text(json.dumps(row))
            self.assertEqual(image_abi.verify_cache(lib,receipt,expected),row)
            lib.write_bytes(b'altered helper')
            with self.assertRaisesRegex(RuntimeError,'differs'):image_abi.verify_cache(lib,receipt,expected)
            lib.write_bytes(b'known helper')
            for field in expected:
                with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'differs'):
                    image_abi.verify_cache(lib,receipt,dict(expected,**{field:'changed'}))

    def test_unknown_abi_refuses_before_any_cache_or_library_access(self):
        with patch.object(image_abi,'require',side_effect=RuntimeError('unverified ABI')),patch.object(image_transport.C,'CDLL') as load:
            with self.assertRaisesRegex(RuntimeError,'unverified ABI'):image_transport.ImageTake()
            load.assert_not_called()
