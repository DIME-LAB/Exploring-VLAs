"""Host selection must not make collection confinement caller-configurable."""
from pathlib import Path
import unittest
from unittest.mock import patch
import host_config


class HostConfigTests(unittest.TestCase):
    def test_known_hosts_and_aliases_select_only_their_root(self):
        for hostname in ('dual-a4500', 'en4226769-l'):
            self.assertEqual(host_config.profile(hostname)['root'], '/mnt/extreme4tb/aaugus11/exploring-vlas')
        for hostname in ('rtx6000', 'EN4226747L'):
            self.assertEqual(host_config.profile(hostname)['root'], '/mnt/storage3/aaugus11/exploring-vlas')

    def test_unknown_host_refuses(self):
        for hostname in ('', 'other-host', '../../rtx6000', 'EN4226747L.evil'):
            with self.subTest(hostname=hostname), self.assertRaises(RuntimeError):
                host_config.profile(hostname)

    def test_each_host_confines_symlinks_and_foreign_roots(self):
        import importlib.util
        for hostname in ('dual-a4500', 'rtx6000'):
            with patch.object(host_config, 'profile', return_value=host_config.profile(hostname)):
                spec = importlib.util.spec_from_file_location('host_storage_test', Path(__file__).with_name('storage.py'))
                storage = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(storage)
            self.assertEqual(storage.confined(storage.CANONICAL_ROOT/'datasets'), storage.CANONICAL_ROOT/'datasets')
            for path in ('/mnt/storage1/ven', '/tmp', storage.CANONICAL_ROOT/'../other'):
                with self.subTest(host=hostname, path=path), self.assertRaises(ValueError):
                    storage.confined(path)
