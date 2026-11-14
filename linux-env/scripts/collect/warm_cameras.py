#!/usr/bin/env python3
"""Warm the assigned card's persistent renderer cache before any recording."""
import argparse,json,os,time
from pathlib import Path
from storage import confined,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--frames',type=int,default=30)
    a=p.parse_args();output=confined(a.output)
    import rclpy
    from sensor_msgs.msg import Image
    rclpy.init();node=rclpy.create_node('collection_camera_cache_warmup');stamps={'wrist':set(),'top':set()}
    first=time.monotonic();subs=[]
    def frame(key,msg):
        if msg.encoding!='rgb8' or (msg.width,msg.height)!=(640,480):raise RuntimeError('Warmup camera layout differs')
        stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        if stamp>0:stamps[key].add(stamp)
    for key,topic in [('wrist','/wrist_camera_rgb_sim'),('top','/workspace_camera_sim')]:
        subs.append(node.create_subscription(Image,topic,lambda msg,key=key:frame(key,msg),64))
    result=dict(passed=False,assigned_gpu_uuid=os.environ['SOARM_ASSIGNED_GPU_UUID'],
                shader_cache=os.environ['SOARM_SHADER_CACHE_DIR'],optix_cache=os.environ['OPTIX_CACHE_PATH'])
    try:
        while min(map(len,stamps.values()))<a.frames:
            if time.monotonic()-first>180:raise TimeoutError('Camera cache warmup did not deliver unique render ticks')
            rclpy.spin_once(node,timeout_sec=.1)
        result.update(passed=True,unique_frames={k:len(v) for k,v in stamps.items()},wall_seconds=time.monotonic()-first)
    finally:
        for key in ('shader_cache','optix_cache'):
            files=[f for f in Path(result[key]).rglob('*') if f.is_file()]
            result[key+'_files']=len(files);result[key+'_bytes']=sum(f.stat().st_size for f in files)
        atomic_json(output,result);node.destroy_node();rclpy.shutdown()
    print(json.dumps(result))


if __name__=='__main__':main()
