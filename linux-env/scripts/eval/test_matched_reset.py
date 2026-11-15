import hashlib,json,tempfile,unittest
from pathlib import Path
from matched_reset_client import require_qualification
class Tests(unittest.TestCase):
    def test_changed_reset_source_cannot_use_a_previous_live_qualification(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'qualification').mkdir();sources=root/'source';sources.mkdir();hashes={}
            for name in ('matched_reset.py','matched_reset_client.py','native_bank.py','qualify_matched_reset.py'):
                (sources/name).write_text(name);hashes[name]=hashlib.sha256(name.encode()).hexdigest()
            host=dict(name='dual-a4500',root='/backup');collector='/backup/collector'
            ledger=dict(status='matched-reset-qualified',cleanup_exit_code=0,host_profile=host,source_sha256=hashes)
            summary=dict(status='PASS',episodes=5,commands=50,bank_sha256='bank',reset_collector_root=collector)
            (root/'window-ledger.json').write_text(json.dumps(ledger));(root/'qualification/summary.json').write_text(json.dumps(summary))
            require_qualification(root,host,collector,'bank',sources)
            (sources/'matched_reset.py').write_text('changed')
            with self.assertRaisesRegex(AssertionError,'differs from live'):require_qualification(root,host,collector,'bank',sources)
if __name__=='__main__':unittest.main()
