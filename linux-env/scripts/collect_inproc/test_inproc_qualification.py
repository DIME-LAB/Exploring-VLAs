import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import shared


class QualificationTests(unittest.TestCase):
    def test_loaded_paths_and_hashes_must_match_the_live_host_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            package=Path(directory)/'package.py';package.write_text('qualified package\n')
            receipt=Path(directory)/'qualification.json'
            record=dict(path=str(package),sha256=hashlib.sha256(package.read_bytes()).hexdigest())
            receipt.write_text(json.dumps(dict(result=dict(version='3.3.2',modules={'charset_normalizer':record}))))
            module=NS(__file__=str(package),__version__='3.3.2')
            with patch.dict(sys.modules,{'charset_normalizer':module}):
                self.assertTrue(shared.require_preloaded_modules(receipt)['passed'])
                package.write_text('unqualified replacement\n')
                with self.assertRaisesRegex(RuntimeError,'differs'):
                    shared.require_preloaded_modules(receipt)
            with patch.dict(sys.modules,{'charset_normalizer':None}):
                with self.assertRaisesRegex(RuntimeError,'not preloaded'):
                    shared.require_preloaded_modules(receipt)


if __name__=='__main__':unittest.main()
