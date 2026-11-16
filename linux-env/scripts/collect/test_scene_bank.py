"""Pure boundary tests; synthetic receipts never represent scored episodes."""
import argparse,ast,copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from scene_bank import SceneBankSession,NAMES,body_poses,pose_errors,configure_args
from storage import CANONICAL_ROOT

class BankTests(unittest.TestCase):
 def setUp(self):
  (CANONICAL_ROOT/'eval').mkdir(exist_ok=True)
  self.tmp=tempfile.TemporaryDirectory(dir=CANONICAL_ROOT/'eval');self.root=Path(self.tmp.name);(self.root/'gen').mkdir();self.reference=self.root/'reference.json';self.reference.write_text(json.dumps(dict(baseline={'red':[0,0,0]},baseline_cup_layout={'mode':'line'})))
  self.args=argparse.Namespace(output=self.root/'gen',scene_bank_output=self.root/'bank',scene_bank_reference_run=self.reference,scene_bank='eval',scene_bank_receipts=None,resume=False,replay_receipt=None,regrasp_only=False,normal_only=False)
 def tearDown(self):self.tmp.cleanup()
 def test_exact_heldout_plan(self):
  s=SceneBankSession(self.args);self.assertEqual([p['seed'] for p in s.plan],list(range(900000,900030)));self.assertEqual([p['color'] for p in s.plan[:3]],['red','green','blue']);self.assertTrue(all(p['split']=='eval' and p['recovery']=='none' for p in s.plan))
 def test_existing_destination_refused(self):
  (self.root/'bank').mkdir()
  with self.assertRaises(FileExistsError):SceneBankSession(self.args)
 def test_partial_cannot_seal(self):
  s=SceneBankSession(self.args)
  with self.assertRaisesRegex(AssertionError,'partial'):s.seal()
 def test_wrong_baseline_refused(self):
  s=SceneBankSession(self.args)
  with self.assertRaisesRegex(AssertionError,'baseline'):s.bind({'red':[.01,0,0]},{'mode':'line'},{}, {})
 def test_pose_shape_and_quaternion_alias(self):
  a={n:dict(position=[0,0,0],orientation=[1,0,0,0],velocity=[0]*6) for n in NAMES};b=body_poses({'bodies':a});c=copy.deepcopy(b)
  for row in c.values():row['orientation']=[-1,0,0,0]
  self.assertEqual(pose_errors(b,c),dict(max_position_m=0,max_quaternion_component=0));self.assertNotIn('velocity',next(iter(b.values())))
  del a[next(iter(NAMES))]
  with self.assertRaises(AssertionError):body_poses({'bodies':a})
 def test_default_args_are_unchanged(self):
  self.args.scene_bank=None;before=vars(self.args).copy();configure_args(self.args,argparse.ArgumentParser());self.assertEqual(vars(self.args),before)
 def test_normal_run_is_an_exact_ast_move(self):
  folder=Path(__file__).parent;reference=json.loads((folder/'fixtures/scene-bank-original-ast.json').read_text());changed=ast.parse((folder/'collect.py').read_text())
  def methods(tree):return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Collector' for n in c.body if isinstance(n,ast.FunctionDef)}
  new=methods(changed)
  class Expand(ast.NodeTransformer):
   def visit_Assign(self,node):
    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_initialize_scene_session':return copy.deepcopy(new['_initialize_scene_session'].body[1:-1])
    return self.generic_visit(node)
   def visit_Expr(self,node):
    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_prepare_scene_attempt':return copy.deepcopy([new['_prepare_scene_attempt'].body[0],*new['_prepare_normal_scene_attempt'].body[1:]])
    return self.generic_visit(node)
  class DefaultPath(ast.NodeTransformer):
   """Project only the explicit opt-in triplet additions out of the normal path."""
   def visit_Assign(self,node):
    if any(isinstance(t,ast.Name) and t.id=='triplet_config' or isinstance(t,ast.Attribute) and t.attr=='triplet_session' for t in node.targets):return None
    return self.generic_visit(node)
   def visit_If(self,node):
    if any(isinstance(v,ast.Name) and v.id=='triplet_config' or isinstance(v,ast.Constant) and v.value=='counterfactual_triplet' for v in ast.walk(node.test)):return None
    return self.generic_visit(node)
   def visit_IfExp(self,node):
    if isinstance(node.body,ast.Call) and isinstance(node.body.func,ast.Name) and node.body.func.id=='triplet_member':return self.visit(node.orelse)
    return self.generic_visit(node)
   def visit_keyword(self,node):
    if node.arg is None and isinstance(node.value,ast.IfExp) and any(isinstance(v,ast.Constant) and v.value=='counterfactual_triplet' for v in ast.walk(node.value.body)):return None
    return self.generic_visit(node)
   def visit_Expr(self,node):
    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='extend' and ast.dump(node.value.args[0])==ast.dump(ast.parse("['linux-env/scripts/collect/scene_identity.py','linux-env/scripts/collect/target_triplets.py']",mode='eval').body):return None
    return self.generic_visit(node)
  def digest(tree):
   tree=copy.deepcopy(tree)
   for node in ast.walk(tree):
    # Python 3.12 adds empty type_params to definitions; the deployed host is 3.10.
    if hasattr(node,'type_params'):
     self.assertEqual(node.type_params,[]);node._fields=tuple(f for f in node._fields if f!='type_params')
   return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
  expanded=DefaultPath().visit(Expand().visit(copy.deepcopy(new['run'])));self.assertEqual(digest(expanded),reference['methods']['run']);self.assertEqual(digest(new['randomize_scene']),reference['methods']['randomize_scene'])
if __name__=='__main__':unittest.main()
