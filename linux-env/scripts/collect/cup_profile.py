"""Conservative cup interior from the full CAD shell, independent of ROS/Isaac."""
import hashlib
import json
import math
from pathlib import Path

PROFILE = json.loads(Path(__file__).with_suffix('.json').read_text())


def radius_at_height(height, profile=PROFILE):
    """Concave envelope: its intersection of frusta is convex, so corners suffice."""
    nodes = profile['samples']
    lines = [(a['height_m'], a['envelope_m'],
              (b['envelope_m']-a['envelope_m'])/(b['height_m']-a['height_m']))
             for a,b in zip(nodes,nodes[1:])]
    return min(radius+slope*(height-z) for z,radius,slope in lines)


def radial_limit(height):
    return radius_at_height(height) - PROFILE['radial_margin_m']


def triangles_from_stl(raw):
    import numpy as np
    import struct
    count = struct.unpack('<I',raw[80:84])[0]
    if len(raw) != 84+50*count:
        raise ValueError('Expected binary STL')
    data = np.frombuffer(raw,offset=84,count=count,
        dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')]))
    return data['v'].astype(float)*.001


def verify_shell(raw, profile=PROFILE):
    """Prove the affine envelope is inside every active nonhorizontal STL plane.

    On each triangle's height interval, both the envelope branch and signed plane
    offset are affine. Endpoints, branch changes and plane zero crossings suffice
    to check the inequality for every intermediate height, not sampled angles.
    Horizontal floor/rim surfaces are handled by the separate height gates.
    """
    import numpy as np
    tri = triangles_from_stl(raw)
    count = len(tri)
    normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    length = np.linalg.norm(normal[:,:2],axis=1)
    offset = -(normal*tri[:,0]).sum(axis=1)
    lo = np.maximum(tri[:,:,2].min(axis=1),.0006)
    hi = np.minimum(tri[:,:,2].max(axis=1),.094)
    active = (hi>lo+1e-12)&(length>1e-14)
    planes = [lo,hi,*[np.full(count,node['height_m']) for node in profile['samples']],
              -offset/np.where(np.abs(normal[:,2])>1e-30,normal[:,2],1e-30)]
    violations = 0
    minimum_clearance = math.inf
    for z in planes:
        valid = active&(z>=lo-1e-12)&(z<=hi+1e-12)
        # Vectorize the same concave affine envelope used at runtime.
        envelope = np.minimum.reduce([a['envelope_m']+(b['envelope_m']-a['envelope_m'])/
            (b['height_m']-a['height_m'])*(z-a['height_m'])
            for a,b in zip(profile['samples'],profile['samples'][1:])])
        distance = np.abs(normal[:,2]*z+offset)/np.maximum(length,1e-30)
        if np.any(valid):
            minimum_clearance = min(minimum_clearance,float(np.min((distance-envelope)[valid])))
        violations += int(np.count_nonzero(valid&(envelope>distance+1e-9)))
    slopes = profile['slopes']
    concave = all(math.isfinite(s) and s>=0 for s in slopes) and all(a>=b for a,b in zip(slopes,slopes[1:]))
    expected_slopes = [(b['envelope_m']-a['envelope_m'])/(b['height_m']-a['height_m'])
                       for a,b in zip(profile['samples'],profile['samples'][1:])]
    concave = concave and all(abs(a-b)<1e-12 for a,b in zip(slopes,expected_slopes))
    source_matches = hashlib.sha256(raw).hexdigest()==profile['source_sha256']
    return dict(passed=not violations and concave and source_matches, triangles=count,
                supporting_plane_violations=violations, concave=concave, source_matches=source_matches,
                min_radial_plane_clearance_m=minimum_clearance)
