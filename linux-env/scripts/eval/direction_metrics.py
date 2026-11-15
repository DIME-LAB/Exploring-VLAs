"""Geometry of a predicted TCP displacement; no claim of closed-loop success."""
import math
import numpy as np

COLORS=('red','green','blue')

def angle_degrees(displacement,bearing,min_norm=1e-12):
    a,b=np.asarray(displacement,dtype=float),np.asarray(bearing,dtype=float)
    if np.linalg.norm(a)<min_norm or np.linalg.norm(b)<min_norm:return None
    cosine=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine,-1,1))))

def motion_directions(start,tcp,blocks,requested,min_motion_mm=1.):
    start=np.asarray(start,dtype=float);tcp=np.asarray(tcp,dtype=float)
    delta=tcp[-1]-start;minimum=min_motion_mm/1000
    xyz_valid=np.linalg.norm(delta)>=minimum;xy_valid=np.linalg.norm(delta[:2])>=minimum
    entries=[]
    for name,position in sorted(blocks.items()):
        colour=name.split('_')[0]
        if colour not in COLORS:continue
        position=np.asarray(position,dtype=float);bearing=position-start
        entries.append(dict(block=name,color=colour,requested=colour==requested,
            angle_3d_deg=angle_degrees(delta,bearing) if xyz_valid else None,
            angle_xy_deg=angle_degrees(delta[:2],bearing[:2]) if xy_valid and np.linalg.norm(bearing[:2])>=minimum else None,
            already_aligned_xy=bool(np.linalg.norm(bearing[:2])<minimum),
            endpoint_distance_mm=float(np.linalg.norm(position-tcp[-1])*1000),
            endpoint_distance_xy_mm=float(np.linalg.norm(position[:2]-tcp[-1,:2])*1000),
            distance_reduction_mm=float((np.linalg.norm(bearing)-np.linalg.norm(position-tcp[-1]))*1000)))
    def classify(key):
        valid=[e for e in entries if e[key] is not None]
        if not valid:return dict(winner_blocks=[],winner_colors=[],requested_winner=None,target_angle_deg=None,other_angle_deg=None,margin_deg=None)
        minimum_angle=min(e[key] for e in valid);winners=[e for e in valid if abs(e[key]-minimum_angle)<1e-6]
        target=[e[key] for e in valid if e['requested']];other=[e[key] for e in valid if not e['requested']]
        colours=sorted({e['color'] for e in winners})
        return dict(winner_blocks=[e['block'] for e in winners],winner_colors=colours,
            requested_winner=(colours==[requested]) if target and len(colours)==1 else None,target_angle_deg=min(target) if target else None,
            other_angle_deg=min(other) if other else None,
            margin_deg=min(other)-min(target) if target and other else None)
    return dict(displacement_world_m=delta.tolist(),motion_mm=float(np.linalg.norm(delta)*1000),
        horizontal_motion_mm=float(np.linalg.norm(delta[:2])*1000),
        path_length_mm=float(np.linalg.norm(np.diff(np.vstack([start,tcp]),axis=0),axis=1).sum()*1000),
        stationary=not xyz_valid,low_horizontal_motion=not xy_valid,
        requested_already_aligned_xy=any(e['requested'] and e['already_aligned_xy'] for e in entries),
        blocks=entries,xyz=classify('angle_3d_deg'),xy=classify('angle_xy_deg'))

def seed_consistency(motions):
    vectors=np.asarray([m['displacement_world_m'] for m in motions]);xy=vectors[:,:2]
    lengths=np.linalg.norm(xy,axis=1);valid=[i for i,m in enumerate(motions) if not m['low_horizontal_motion']]
    units=np.asarray([xy[i]/lengths[i] for i in valid])
    pair_angles=[angle_degrees(units[i],units[j]) for i in range(len(units)) for j in range(i+1,len(units))]
    endpoint_spread=np.linalg.norm(vectors-vectors.mean(axis=0),axis=1)*1000
    requested=[m['xy']['requested_winner'] for m in motions]
    return dict(n=len(motions),directional_xy_n=len(valid),low_horizontal_n=len(motions)-len(valid),
        requested_direction_wins=sum(v is True for v in requested),
        requested_win_denominator=sum(v is not None for v in requested),
        wrong_colour_n=sum(v is False for v in requested),
        ambiguous_colour_n=sum(not m['low_horizontal_motion'] and len(m['xy']['winner_colors'])>1 for m in motions),
        undefined_requested_bearing_n=sum(not m['low_horizontal_motion'] and len(m['xy']['winner_colors'])<=1 and m['xy']['requested_winner'] is None for m in motions),
        small_margin_under5deg_n=sum(m['xy']['margin_deg'] is not None and abs(m['xy']['margin_deg'])<5 for m in motions),
        requested_winner_distinct_blocks=sorted({b for m in motions if m['xy']['requested_winner'] is True for b in m['xy']['winner_blocks']}),
        direction_resultant=float(np.linalg.norm(units.mean(axis=0))) if len(units) else None,
        magnitude_weighted_coherence=float(np.linalg.norm(xy.mean(axis=0))/lengths.mean()) if lengths.mean()>0 else None,
        max_seed_direction_angle_deg=max(pair_angles) if pair_angles else None,
        endpoint_spread_rms_mm=float(np.sqrt(np.mean(endpoint_spread**2))),
        mean_displacement_world_m=vectors.mean(axis=0).tolist(),
        winner_block_sets=[m['xy']['winner_blocks'] for m in motions])
