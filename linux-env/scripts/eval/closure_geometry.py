"""CONFIRMED-closure geometry from actual TCP/body axes, never wrist-roll yaw."""
import math
import numpy as np

def confirmed_geometry(snapshot,object_name):
    telemetry=snapshot.get('grasp_telemetry') or {}
    if 'tcp_world_matrix' not in telemetry:
        return dict(status='unavailable',phase='CONFIRMED',reason='native TCP matrix unavailable',object=object_name)
    matrix=np.asarray(telemetry['tcp_world_matrix'],dtype=float)
    position=np.asarray(snapshot['bodies'][object_name]['position'],dtype=float)
    q=np.asarray(snapshot['bodies'][object_name]['orientation'],dtype=float)
    if matrix.shape!=(4,4) or position.shape!=(3,) or q.shape!=(4,) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(q)) or np.linalg.norm(q)==0:
        raise ValueError('Invalid native closure pose')
    w,x,y,z=q/np.linalg.norm(q)
    block_x=np.asarray([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)])
    rotation=matrix[:3,:3];tcp=matrix[:3,3];jaw_y=rotation[:,1]
    jaw_xy=float(np.linalg.norm(jaw_y[:2]));block_xy=float(np.linalg.norm(block_x[:2]))
    yaw=None if min(jaw_xy,block_xy)<1e-9 else (math.degrees(math.atan2(jaw_y[1],jaw_y[0])-math.atan2(block_x[1],block_x[0]))+90)%180-90
    delta=tcp-position;relative=rotation.T@(position-tcp)
    return dict(status='available',phase='CONFIRMED',trigger='first measured >=2deg gripper close',
                object=object_name,epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],
                xy_mm=float(np.linalg.norm(delta[:2])*1000),dz_mm=float(delta[2]*1000),
                tcp_minus_block_world_mm=(delta*1000).tolist(),block_in_tcp_frame_mm=(relative*1000).tolist(),
                yaw_deg=yaw,yaw_status='unavailable_degenerate_horizontal_axis' if yaw is None else 'available',
                jaw_y_tilt_deg=math.degrees(math.atan2(abs(jaw_y[2]),jaw_xy)),
                block_x_tilt_deg=math.degrees(math.atan2(abs(block_x[2]),block_xy)),
                yaw_definition='world-horizontal TCP local+Y vs block local+X, modulo180deg; not full3D alignment',
                tcp_definition='actual fixed-jaw tip; not aperture centre')
