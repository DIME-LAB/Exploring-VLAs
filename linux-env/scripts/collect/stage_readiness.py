"""Read-only stage admission checks for the collector's exact baseline query."""
import ast,hashlib,math,time
from pathlib import Path


def baseline_code():
    source=ast.parse(Path(__file__).with_name('collect.py').read_text());found=[]
    for node in ast.walk(source):
        if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='initial' for t in node.targets):continue
        if not isinstance(node.value,ast.Call):continue
        try:params=ast.literal_eval(node.value.args[1])
        except (ValueError,IndexError,TypeError):continue
        if not isinstance(params,dict):continue
        code=params.get('code','')
        if isinstance(code,str) and 'get_instance, CUP_LAYOUT' in code:found.append(code)
    if len(found)!=1:raise RuntimeError('Collector baseline query is not uniquely identified')
    return found[0]


def traced_baseline(code):
    class Trace(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and
                    node.func.value.id=='st' and node.func.attr=='GetPrimAtPath'):
                return ast.copy_location(ast.Call(func=ast.Name(id='_trace_prim',ctx=ast.Load()),args=node.args,keywords=node.keywords),node)
            return node
    helper="def _trace_prim(path):\n p=st.GetPrimAtPath(path)\n _trace.append({'path':path,'valid':bool(p and p.IsValid())})\n return p\n"
    return helper+ast.unparse(ast.fix_missing_locations(Trace().visit(ast.parse(code))))


def probe_code():
    legacy=baseline_code()
    prefix="""import collection_runtime as c,omni.usd,time
from so_arm101_dt.extension import get_instance
from pxr import Usd,UsdGeom
context=omni.usd.get_context();stage=context.get_stage();r=c.runtime();instance=get_instance()
# Isaac 6.1 stage_utils.is_stage_loading uses the third status entry.
status=list(context.get_stage_loading_status())
proof={'version':1,'wall_ns':time.time_ns(),'loading_status':status,'stage_root':stage.GetRootLayer().identifier if stage else None,'rows':[],'missing_required':[],'errors':[],'legacy_baseline_passed':False}
required=['/World/SO_ARM101','/World/SO_ARM101/root_joint','/World/SO_ARM101/base_link','/World/workspace_camera',*r.body_paths.values()]
for path in required:
 if stage is None or not stage.GetPrimAtPath(path).IsValid():proof['missing_required'].append(path)
root=stage.GetPrimAtPath('/World/Objects') if stage else None
proof['objects_root_valid']=bool(root and root.IsValid())
if proof['objects_root_valid']:
 for child in root.GetChildren():
  name=child.GetName();row={'name':name,'excluded':name=='PhysicsMaterial','wrapper':str(child.GetPath()),'type':child.GetTypeName()}
  if not row['excluded']:
   try:
    path=instance._get_prim_path(name);row['path']=path;p=stage.GetPrimAtPath(path)
    row['valid']=bool(p and p.IsValid());row['xformable']=bool(UsdGeom.Xformable(p)) if row['valid'] else False
    row['matches_native_body']=r.body_paths.get(name)==path
    if row['xformable']:row['position']=list(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
   except Exception as error:row['error']=repr(error)
  proof['rows'].append(row)
proof['expected_objects']=sorted(name for name in r.body_paths if not name.startswith('cup_'))
if proof['objects_root_valid'] and not proof['missing_required'] and len(status)==3 and status[2]==0:
 try:
  legacy_scope={}
"""
    suffix="""  proof['legacy_baseline_passed']=True
  proof['baseline_object_names']=sorted(legacy_scope['result']['objects'])
 except Exception as error:
  proof['errors'].append({'field':'exact_collector_baseline_query','error':repr(error)})
result=proof
"""
    diagnostic="  proof['legacy_query_trace']=[]\n  try:exec("+repr(traced_baseline(legacy))+",{'_trace':proof['legacy_query_trace']})\n  except Exception as trace_error:proof['legacy_trace_error']=repr(trace_error)\n"
    return prefix+'  exec('+repr(legacy)+',legacy_scope)\n'+suffix.replace('result=proof',diagnostic+'result=proof')


def errors(proof):
    try:
        issues=[]
        if proof['version']!=1 or not proof['stage_root'] or not proof['objects_root_valid']:issues.append('stage/root is unavailable')
        if len(proof['loading_status'])!=3 or proof['loading_status'][2]!=0:issues.append('stage assets still loading')
        issues.extend('missing required prim '+p for p in proof['missing_required'])
        rows=[r for r in proof['rows'] if not r['excluded']]
        names=[r['name'] for r in rows]
        if len(proof['expected_objects'])!=6 or sorted(names)!=proof['expected_objects']:issues.append('baseline child set differs from six native objects')
        for row in rows:
            if not all(row.get(k) for k in ('valid','xformable','matches_native_body')) or row.get('error'):
                issues.append('unready baseline prim '+row.get('path',row['wrapper'])+': '+str(row.get('error','invalid schema/body mapping')))
            elif len(row.get('position',[]))!=3 or not all(math.isfinite(x) for x in row['position']):issues.append('baseline transform unreadable '+row['path'])
        if not proof['legacy_baseline_passed'] or proof.get('baseline_object_names')!=proof['expected_objects']:issues.append('exact collector baseline query is not ready')
        issues.extend('exact baseline accessed invalid prim '+row['path'] for row in proof.get('legacy_query_trace',[]) if not row['valid'])
        issues.extend(str(e) for e in proof['errors'])
        return issues
    except (KeyError,TypeError,ValueError):return ['stage readiness proof is incomplete']


def wait(read,publish,timeout_s=120.,interval_s=.25,stable_samples=3,clock=time.monotonic,pause=time.sleep):
    if timeout_s<=0 or interval_s<=0 or stable_samples<1:raise ValueError('Invalid stage readiness bounds')
    start=clock();records=[];consecutive=0;root=None
    while clock()-start<timeout_s:
        observation=read();proof=observation['stage_readiness'];problems=errors(proof)
        if clock()-start>=timeout_s:problems.append('readiness reply exceeded its deadline')
        same_root=root is None or root==proof.get('stage_root')
        consecutive=consecutive+1 if not problems and same_root else 1 if not problems else 0
        root=proof.get('stage_root');records.append(dict(observation=observation,errors=problems))
        report=dict(version=1,passed=consecutive>=stable_samples,wall_s=clock()-start,timeout_s=timeout_s,stable_samples=stable_samples,
            legacy_code_sha256=hashlib.sha256(baseline_code().encode()).hexdigest(),samples=records)
        publish(report)
        if report['passed']:return report
        pause(min(interval_s,max(0.,timeout_s-(clock()-start))))
    raise TimeoutError('Stage readiness deadline: '+repr(records[-1]['errors'] if records else ['no response']))
