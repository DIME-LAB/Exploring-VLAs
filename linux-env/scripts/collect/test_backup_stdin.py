"""Real backup recheck with a manifest larger than Linux MAX_ARG_STRLEN."""
import json,os,shlex,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import triplet_continuation as c
import triplet_production as p
import triplet_artifacts as a
from backup_space import checked_script_payload,check_script_receiver_code


class BackupStdinTests(unittest.TestCase):
    def remote(self,mutate=lambda data:data):
        real=subprocess.check_output
        def run(argv,**kwargs):
            self.assertLess(len(argv[-1].encode()),4096)
            self.assertIn('input',kwargs);self.assertIsInstance(kwargs['input'],bytes)
            kwargs['input']=mutate(kwargs['input']);kwargs['stderr']=subprocess.PIPE
            return real([sys.executable,'-B','-c',shlex.split(argv[-1])[2]],**kwargs)
        return run

    def test_real_backup_recheck_accepts_manifest_over_one_mib_without_argv_payload(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i in range(4096):(root/(str(i).zfill(6)+'-'+'x'*210)).write_bytes(str(i).encode())
            files=a.manifest(root);self.assertGreater(len(json.dumps(files).encode()),1024**2)
            proof=dict(files_sha256=files,backup=dict(destination=str(root)))
            with patch.object(p,'DUAL',root),patch.object(p.subprocess,'check_output',self.remote()):
                self.assertEqual(c.backup_recheck(proof),dict(passed=True,files=4096))
                (root/'extra').write_bytes(b'not manifested')
                with self.assertRaises(subprocess.CalledProcessError):c.backup_recheck(proof)

    def test_real_backup_recheck_refuses_truncated_extra_and_hash_changed_input(self):
        def altered_hash(data):
            head,body=data.split(b'\n',1);value=json.loads(head);value['sha256']='0'*64
            return json.dumps(value).encode()+b'\n'+body
        variants=[('truncated',lambda data:data[:-1],b'input truncated'),('extra',lambda data:data+b'\n',b'extra bytes'),('sha',altered_hash,b'input SHA differs')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'file').write_bytes(b'source');proof=dict(files_sha256=a.manifest(root),backup=dict(destination=str(root)))
            for kind,mutate,error in variants:
                with self.subTest(kind=kind),patch.object(p,'DUAL',root),patch.object(p.subprocess,'check_output',self.remote(mutate)):
                    with self.assertRaises(subprocess.CalledProcessError) as failure:c.backup_recheck(proof)
                    self.assertEqual(failure.exception.output,b'');self.assertIn(error,failure.exception.stderr)

    def test_framing_is_checked_before_any_program_effect(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'executed';wire=checked_script_payload('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
            for bad in (wire[:-1],wire+b'\n'):
                result=subprocess.run([sys.executable,'-B','-c',check_script_receiver_code()],input=bad,capture_output=True)
                self.assertNotEqual(result.returncode,0);self.assertFalse(marker.exists())
            result=subprocess.run([sys.executable,'-B','-c',check_script_receiver_code()],input=wire,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(marker.exists())


if __name__=='__main__':unittest.main()
