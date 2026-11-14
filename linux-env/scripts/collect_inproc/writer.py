"""Bounded RGB writer producing the existing collector's raw attempt layout."""
import copy
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
import shared
from storage import confined
from core import image_identity


class Writer:
    def __init__(self,path,render=True):
        self.path=confined(path);self.path.mkdir(parents=True,exist_ok=False)
        self.rows=[];self.profiles=[];self.queue=queue.Queue(maxsize=8)
        self.failure=None;self.encoders={};self.logs=[]
        if render:
            try:
                for key in ('wrist','top'):
                    log=(self.path/(key+'-encode.log')).open('w');self.logs.append(log)
                    self.encoders[key]=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y',
                        '-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','30','-i','-',
                        '-an','-c:v','libx264','-preset','ultrafast','-crf','18','-threads','1',
                        '-pix_fmt','yuv420p',str(self.path/(key+'.mp4'))],
                        stdin=subprocess.PIPE,stderr=log,start_new_session=True)
            except BaseException:
                self.shutdown_encoders(abort=True)
                raise
        self.thread=threading.Thread(target=self.work,name='inproc-encoder',daemon=True)
        self.thread.start()

    def work(self):
        try:
            with (self.path/'frames.jsonl').open('w') as frames,(self.path/'poses.jsonl').open('w') as poses:
                while True:
                    item=self.queue.get()
                    if item is None:break
                    row,images,pose,profile,enqueued=item;start=time.perf_counter_ns()
                    if set(images or {})!=set(self.encoders):raise ValueError('Encoder camera inventory differs')
                    for key,process in self.encoders.items():process.stdin.write(images[key].tobytes())
                    frames.write(json.dumps(row,allow_nan=False)+'\n')
                    poses.write(json.dumps(pose,allow_nan=False)+'\n')
                    self.rows.append(row)
                    self.profiles.append(dict(profile,queue_ms=(start-enqueued)/1e6,
                        write_ms=(time.perf_counter_ns()-start)/1e6))
                frames.flush();poses.flush()
        except BaseException as exc:self.failure=exc

    def add(self,row,images,pose,profile):
        if self.failure:raise RuntimeError('Encoder failed') from self.failure
        if set(images or {})!=set(self.encoders):raise ValueError('Encoder camera inventory differs')
        for img in (images or {}).values():image_identity(img,row['sim_ns'])
        # Arrays are owned copies from the annotators; no next-frame overwrite.
        start=time.perf_counter_ns()
        self.queue.put((copy.deepcopy(row),images,pose,profile,start),timeout=5)
        return (time.perf_counter_ns()-start)/1e6

    def shutdown_encoders(self,abort=False):
        errors=[]
        for p in self.encoders.values():
            try:
                if abort and p.poll() is None:p.terminate()
                if p.stdin:p.stdin.close()
                code=p.wait(timeout=30)
                if code:errors.append('ffmpeg exit '+str(code))
            except Exception as exc:
                errors.append(repr(exc))
                if p.poll() is None:p.kill();p.wait(timeout=5)
        for log in self.logs:log.close()
        if errors and not abort:raise RuntimeError('; '.join(errors))

    def close(self):
        try:
            if self.thread.is_alive():self.queue.put(None,timeout=5)
            self.thread.join(timeout=30)
            if self.thread.is_alive():raise RuntimeError('Encoder did not drain')
            if self.failure:raise RuntimeError('Encoder failed') from self.failure
            self.shutdown_encoders()
        except BaseException:
            self.shutdown_encoders(abort=True)
            self.thread.join(timeout=5)
            raise
