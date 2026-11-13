"""Calibrate a conservative concave radial envelope against the full cup STL."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from cup_profile import PROFILE,triangles_from_stl,verify_shell
from storage import confined
repo=Path(__file__).resolve().parents[3]
source=repo/'vla_SO-ARM101/src/so_arm101_description/meshes/cup/cup.stl'
tri=triangles_from_stl(source.read_bytes())
# First node is 10 nm above the authored floor to avoid a coplanar slice.
nodes=np.array([.00120001,.002,.004,.094])
def segments(z):
 t=tri[(tri[:,:,2].min(axis=1)<z)&(tri[:,:,2].max(axis=1)>=z)]
 found=[]
 for a,b in ((0,1),(1,2),(2,0)):
  p=t[:,a];q=t[:,b];valid=((p[:,2]<z)&(q[:,2]>=z))|((q[:,2]<z)&(p[:,2]>=z))
  vals=p[valid]+(z-p[valid,2])[:,None]/(q[valid,2]-p[valid,2])[:,None]*(q[valid]-p[valid])
  found.append((np.nonzero(valid)[0],vals))
 by={}
 for ids,vs in found:
  for i,v in zip(ids,vs):by.setdefault(int(i),[]).append(v[:2])
 s=np.array([v for v in by.values() if len(v)==2]);a=s[:,0];e=s[:,1]-a
 ratio=np.clip(-(a*e).sum(axis=1)/np.maximum((e*e).sum(axis=1),1e-30),0,1)
 return np.linalg.norm(a+ratio[:,None]*e,axis=1).min()
radii=np.array([segments(z) for z in nodes])
fit_margin=.0001
normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal[:,:2],axis=1)
d=-(normal*tri[:,0]).sum(axis=1)
support=[]
for z in nodes:
 active=(tri[:,:,2].min(axis=1)<=z)&(tri[:,:,2].max(axis=1)>=z)&(length>1e-14)
 support.append(np.min(np.abs(normal[active,2]*z+d[active])/length[active]))
bound=np.minimum(radii,np.array(support))-fit_margin
slopes=np.diff(bound)/np.diff(nodes)
assert np.all(np.diff(slopes)<=0),slopes
def radius(z):
 idx=np.clip(np.searchsorted(nodes,z,side='right')-1,0,len(nodes)-2)
 return bound[idx]+slopes[idx]*(z-nodes[idx])
normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal[:,:2],axis=1)
d=-(normal*tri[:,0]).sum(axis=1)
lo=np.maximum(tri[:,:,2].min(axis=1),.0006);hi=np.minimum(tri[:,:,2].max(axis=1),.094)
active=(hi>lo+1e-12)&(length>1e-14)
violations=[]
for z in [lo,hi,*[np.full(len(tri),z) for z in nodes],-d/np.where(np.abs(normal[:,2])>1e-30,normal[:,2],1e-30)]:
 valid=active&(z>=lo-1e-12)&(z<=hi+1e-12)
 plane=np.abs(normal[:,2]*z+d)/np.maximum(length,1e-30)
 gap=radius(z)-plane
 violations.extend((float(gap[i]),int(i),float(z[i])) for i in np.flatnonzero(valid & (gap>1e-9)))
report=dict(source=str(source.relative_to(repo)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 method='Minimum axis distance of each STL/plane slice; concave four-node chord envelope, 100 um subtraction at nodes; each active triangle supporting-plane inequality checked at all affine interval endpoints and zero crossings',
 samples=[dict(height_m=float(z),measured_inradius_m=float(r),supporting_plane_inradius_m=float(t),envelope_m=float(b)) for z,r,t,b in zip(nodes,radii,support,bound)],
 slopes=list(map(float,slopes)),fit_allowance_m=fit_margin,radial_margin_m=.0007,
 radial_margin_basis=PROFILE['radial_margin_basis'],
 effective_offsets_m=PROFILE['effective_offsets_m'],effective_offsets_source=PROFILE['effective_offsets_source'],
 triangle_count=len(tri),supporting_plane_violations=len(violations),worst_violations=sorted(violations,reverse=True)[:10])
assert verify_shell(source.read_bytes(),report)['passed'], report
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,help='Optional calibration report under the collection output root')
args=parser.parse_args()
if args.output:confined(args.output).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
