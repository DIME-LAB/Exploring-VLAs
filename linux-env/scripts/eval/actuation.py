"""Opt-in SIM eval immediate waypoints through the unchanged native backend."""
def install(robot,mode,use_sim_time,witness=None):
    if mode not in ('native','immediate'):raise ValueError('Unknown eval actuation mode')
    if not use_sim_time or not robot.config.actuate or not robot.config.use_degrees:raise ValueError('Actuation replay requires sim time and native actuated degree features')
    originals=[];original_duration=robot.config.action_duration_s
    if mode=='immediate':robot.config.action_duration_s=0.
    try:
        for label in ('arm','gripper'):
            pub=getattr(robot,'_'+label+'_command_pub');original=pub.publish;originals.append((pub,original))
            def publish(message,original=original,label=label):
                if len(message.points)!=1:raise ValueError('Eval actuation requires a single native waypoint')
                if mode=='immediate' and (message.header.stamp.sec or message.header.stamp.nanosec or message.points[0].time_from_start.sec or message.points[0].time_from_start.nanosec):raise ValueError('Native backend did not emit an immediate start-now waypoint')
                if witness is not None:witness(label,message)
                return original(message)
            pub.publish=publish
    except BaseException:
        robot.config.action_duration_s=original_duration
        for pub,original in originals:pub.publish=original
        raise
    def restore():
        robot.config.action_duration_s=original_duration
        for pub,original in originals:pub.publish=original
    return restore
