"""Launcher ownership refuses unrecorded/reused PIDs without opening a stop channel."""
import json,unittest
from unittest.mock import Mock,patch
import managed_process


class ManagedProcessTests(unittest.TestCase):
    def setUp(self):
        self.reader=patch('managed_process.read_leaf',side_effect=lambda path:path.read_text());self.reader.start()
        self.addCleanup(self.reader.stop)

    def fixture(self,receipt=None,current=None):
        path=Mock();path.exists.return_value=True;path.read_text.return_value='41234'
        record=Mock();record.exists.return_value=receipt is not None
        record.read_text.return_value=json.dumps(receipt)
        return path,record,current or dict(pid=41234,start_ticks='current',pgid=41234,state='S')

    def test_foreign_live_pid_without_receipt_refuses_status_and_stop(self):
        path,record,current=self.fixture()
        with patch('managed_process.record_path',return_value=record),patch('managed_process.identity',return_value=current),\
             patch('managed_process.socket.socket') as network:
            with self.assertRaisesRegex(RuntimeError,'without a launcher identity receipt'):managed_process.verified(path)
            with self.assertRaisesRegex(RuntimeError,'without a launcher identity receipt'):managed_process.stop(path)
            network.assert_not_called()

    def test_reused_pid_or_changed_group_refuses_before_any_signal(self):
        receipt=dict(supervisor=dict(pid=41234,start_ticks='old',pgid=41234),state='running')
        path,record,current=self.fixture(receipt)
        with patch('managed_process.record_path',return_value=record),patch('managed_process.identity',return_value=current),\
             patch('managed_process.os.pidfd_open',create=True) as handle:
            with self.assertRaisesRegex(RuntimeError,'foreign/reused'):managed_process.stop(path)
            handle.assert_not_called()

    def test_missing_old_pid_is_already_down_and_never_signalled(self):
        path,record,current=self.fixture(dict(supervisor=dict(pid=41234,start_ticks='old',pgid=41234),state='stopped'))
        with patch('managed_process.record_path',return_value=record),patch('managed_process.identity',return_value=None):
            self.assertEqual(managed_process.stop(path),dict(stopped=True,already_down=True))
