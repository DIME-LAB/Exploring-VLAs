"""A growing historical log never becomes a mutable renderer qualification."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import bootstrap_capsule as capsule
from renderer_device import parse_table,resolve
from triplet_artifacts import sha


TABLE='''| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |
| 0 | NVIDIA RTX 6000 Ada | Yes:0 | 0 | 49140 | 10de | 0 |
| | | | | | | c2772797.. |
| | | | | | AC | |
| end |
'''
CARDS=[dict(index=0,uuid=capsule.ASSIGNED_UUID,pci_bus_id='00000000:AC:00.0')]


class BootstrapCapsuleTests(unittest.TestCase):
    def fixture(self,d):
        root=Path(d);source=root/'historical-kit.log';source.write_text(TABLE)
        digest=hashlib.sha256(TABLE.encode()).hexdigest()
        pin=resolve(parse_table(TABLE),CARDS,capsule.ASSIGNED_UUID,require_active=True)
        pin['source']=dict(path=str(source),sha256=digest);path=root/'old-pin.json';path.write_text(json.dumps(pin))
        return source,path,digest,root/'immutable'
    def test_exact_qualified_prefix_survives_original_log_append_and_is_shared_by_slots(self):
        with tempfile.TemporaryDirectory() as d:
            source,pin,digest,out=self.fixture(d);source.write_text(TABLE+'late ROS shutdown output\n');before=source.read_bytes()
            with patch.object(capsule,'QUALIFIED_LOG_SHA256',digest):
                result=capsule.create(pin,out,cards=CARDS)
                self.assertEqual(sha(out/'kit-bootstrap.log'),digest)
                self.assertEqual((out/'kit-bootstrap.log').read_bytes(),TABLE.encode())
                source.write_bytes(before+b'more unrelated shutdown lines\n')
                for slot in (3,4,5,6):
                    loaded=capsule.load(out/'capsule.json',result['capsule_sha256'],cards=CARDS)
                    self.assertEqual(loaded['pin']['source']['sha256'],digest)
                    self.assertEqual(loaded['pin']['source']['path'],str(out/'kit-bootstrap.log'))
                self.assertEqual((out/'kit-bootstrap.log').read_bytes(),TABLE.encode())
                self.assertEqual((out/'kit-bootstrap.log').stat().st_mode&0o222,0)
                self.assertEqual(json.loads(pin.read_text())['source']['path'],str(source))
    def test_changed_original_prefix_refuses_instead_of_requalifying_a_new_hash(self):
        with tempfile.TemporaryDirectory() as d:
            source,pin,digest,out=self.fixture(d);source.write_text(TABLE.replace('Yes:0','No:0')+'tail')
            with patch.object(capsule,'QUALIFIED_LOG_SHA256',digest):
                with self.assertRaisesRegex(ValueError,'no exact SHA-matching'):capsule.create(pin,out,cards=CARDS)
            self.assertFalse(out.exists())
    def test_fixture_append_capsule_edit_or_current_device_change_refuses(self):
        for kind in ('fixture','capsule','device','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                source,pin,digest,out=self.fixture(d)
                with patch.object(capsule,'QUALIFIED_LOG_SHA256',digest):
                    result=capsule.create(pin,out,cards=CARDS);cards=copy.deepcopy(CARDS)
                    if kind=='fixture':
                        p=out/'kit-bootstrap.log';p.chmod(0o600);p.write_text(TABLE+'late output')
                    elif kind=='capsule':
                        p=out/'capsule.json';p.chmod(0o600);v=json.loads(p.read_text());v['qualified_pin']['kit_index']=1;p.write_text(json.dumps(v))
                    elif kind=='symlink':
                        p=out/'kit-bootstrap.log';p.unlink();p.symlink_to(source)
                    else:cards[0]['pci_bus_id']='00000000:AD:00.0'
                    with self.assertRaises((ValueError,RuntimeError)):
                        capsule.load(out/'capsule.json',result['capsule_sha256'],cards=cards)
    def test_actual_plan_binds_one_capsule_and_worker_refuses_a_late_capsule_change(self):
        import triplet_production as production
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,pin,digest,_=self.fixture(d);out=root/'setup/counterfactual-kit-bootstrap-v1'
            with patch.object(capsule,'QUALIFIED_LOG_SHA256',digest),patch.object(production,'ROOT',root),\
                    patch.object(production,'require_host'),patch.object(production,'frozen'):
                created=capsule.create(pin,out,cards=CARDS);recipe=production.plan('a'*40)
                self.assertEqual(recipe['bootstrap_capsule_sha256'],created['capsule_sha256'])
                self.assertEqual(recipe['bootstrap_capsule_path'],str(out/'capsule.json'))
                recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe))
                p=out/'capsule.json';p.chmod(0o600);value=json.loads(p.read_text());value['recovery']['appended_bytes']+=1;p.write_text(json.dumps(value))
                with patch.object(sys,'argv',['triplet_production.py','--worker','3','--wave','0','--recipe',str(recipe_path)]),\
                        patch.object(production.subprocess,'Popen') as spawn:
                    with self.assertRaisesRegex(ValueError,'Worker recipe changed'):production.main()
                    spawn.assert_not_called()


if __name__=='__main__':unittest.main()
