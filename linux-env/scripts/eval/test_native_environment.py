import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from native_environment import policy_python
class Tests(unittest.TestCase):
    def test_venv_entry_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);base=root/'base-python';base.write_text('binary')
            entry=root/'venv/bin/python';entry.parent.mkdir(parents=True);entry.symlink_to(base)
            self.assertEqual(policy_python({'name':'dual-a4500'},entry),str(entry))
            self.assertNotEqual(policy_python({'name':'dual-a4500'},entry),str(base))
    def test_dual_does_not_guess_an_interpreter(self):
        with patch.dict(os.environ,{'SOARM_EVAL_POLICY_PYTHON':''}):
            with self.assertRaisesRegex(ValueError,'qualified Torch interpreter'):policy_python({'name':'dual-a4500'})
if __name__=='__main__':unittest.main()
