"""Executed inside the assigned Kit process by the collection-rate launcher command."""
import carb
import omni.usd
import omni.graph.core as og
import omni.syntheticdata
from isaacsim.core.api import World
from pxr import Sdf, UsdGeom


def configure(fps, preset="collect", rate_cap=0):
    if fps not in (20, 30):
        raise ValueError('Collection fps must be 20 or 30')
    world = World.instance()
    if world is None:
        raise RuntimeError('Create the World before configuring the collection clock')
    if preset not in ('collect', 'deploy'):
        raise ValueError('Preset must be collect or deploy')
    # Controllers and MoveIt use /clock; cameras publish every 30 Hz sim tick.
    if abs(world.get_physics_dt() - 1 / 120) > 1e-8 or abs(omni.usd.get_context().get_stage().GetTimeCodesPerSecond() - 30) > 1e-8:
        raise RuntimeError('Collection requires a fresh World with physics_dt=1/120 and rendering_dt=1/30; never change World timing while live')
    stage = omni.usd.get_context().get_stage()
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            if not prim.HasAPI('OmniSensorAPI'):
                prim.ApplyAPI('OmniSensorAPI')
            attr = prim.GetAttribute('omni:sensor:tickRate')
            if not attr or attr.Get() != float(fps):
                prim.CreateAttribute('omni:sensor:tickRate', Sdf.ValueTypeNames.Float).Set(float(fps))
            cameras.append(str(prim.GetPath()))
    camera_gates = []
    for graph in ('ActionGraph_WristCamera', 'ActionGraph_WorkspaceCamera'):
        rp = og.Controller.attribute('/Graph/' + graph + '/RGBPublish.inputs:renderProductPath').get()
        # This 6.1 viewport-backed path ticks the writer every app frame even when the
        # sensor is set to 30 Hz. This app-frame gate does not suppress paused re-renders;
        # the collector selects one packet per publisher/sim stamp using RMW identity.
        omni.syntheticdata.SyntheticData.Get().set_node_attributes(
            'LdrColorSDIsaacSimulationGate', {'inputs:step': 30 // fps}, rp)
        camera_gates.append(str(rp))
    settings = carb.settings.get_settings()
    settings.set_bool('/app/player/useFixedTimeStepping', True)
    settings.set_bool('/app/runLoops/main/syncToPresent', False)
    settings.set_bool('/app/runLoops/main/rateLimitUseBusyLoop', False)
    if settings.get('/persistent/simulation/minFrameRate') != 30:
        raise RuntimeError('Launch the scene with minFrameRate=30 so physics advances four 120 Hz substeps per app frame')
    cap = 30 if preset == 'deploy' else rate_cap
    settings.set_bool('/app/runLoops/main/rateLimitEnabled', cap > 0)
    settings.set_int('/app/runLoops/main/rateLimitFrequency', int(cap))
    settings.set_bool('/persistent/omni/replicator/captureOnPlay', True)
    return dict(fps=fps, physics_dt=world.get_physics_dt(), min_frame_rate=settings.get('/persistent/simulation/minFrameRate'), rendering_dt=1.0 / stage.GetTimeCodesPerSecond(),
                rate_limit=settings.get('/app/runLoops/main/rateLimitFrequency'), cameras=cameras, camera_gates=camera_gates,
                time_codes_per_second=stage.GetTimeCodesPerSecond(), preset=preset, rate_cap=cap, target_rtf=1.0 if preset == 'deploy' else None)
