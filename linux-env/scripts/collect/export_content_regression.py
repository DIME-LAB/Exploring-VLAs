"""CPU Parquet regressions; run with the dataset (pixi) environment."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from export_content import validate_metadata,ContentMismatch
from export_video import validate_videos,compare_rgb


class ExportContentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.run=Path(self.tmp.name);root=self.run/'dataset'
        (root/'meta/episodes').mkdir(parents=True);(root/'data').mkdir()
        def js(path,value):path.write_text(json.dumps(value))
        js(self.run/'summary.json',dict(complete=True,end_marker='COLLECTION_FINISHED',episodes=2))
        js(self.run/'collector-finalization.json',dict(completed=True,passed=True,cleanup_errors=[]))
        manifest=[];self.data=[];meta=[];tasks=[]
        for i,color in enumerate(('red','green')):
            task=f'Pick a {color} lego and place it in {color} cup';tasks.append(dict(task_index=i,task=task))
            receipt=dict(accepted=True,episode_id='ep'+str(i),params=dict(seed=600000+i,task=task))
            manifest.append(dict(episode_index=i,**receipt));attempt=self.run/f'attempt-{i:04}';attempt.mkdir();js(attempt/'receipt.json',receipt)
            rows=[]
            for j in range(3):
                stamp=10**9+i*10**9+j*33333333;state=[.1*j]*6;action=[.02*j]*6
                rows.append(dict(episode_id=receipt['episode_id'],sim_ns=stamp,state=state,action=action))
                self.data.append(dict(timestamp=float(np.float32(j*.033333333)),frame_index=j,episode_index=i,index=i*3+j,task_index=i,
                    **{'observation.state':np.rad2deg(np.asarray(state,np.float32)).tolist(),'action':np.rad2deg(np.asarray(action,np.float32)).tolist()}))
            (attempt/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            meta.append(dict(episode_index=i,tasks=[task],length=3,dataset_from_index=i*3,dataset_to_index=i*3+3))
        js(root/'meta/info.json',dict(fps=30,total_episodes=2,total_frames=6));js(root/'meta/collection_episodes.json',manifest)
        pq.write_table(pa.Table.from_pylist(tasks),root/'meta/tasks.parquet');pq.write_table(pa.Table.from_pylist(meta),root/'meta/episodes/file-000.parquet')
        self.root=root;self.save(self.data)
    def tearDown(self):self.tmp.cleanup()
    def save(self,rows):pq.write_table(pa.Table.from_pylist(rows),self.root/'data/file-000.parquet')
    def test_complete_dataset_passes_all_rows(self):
        result=validate_metadata(self.run);self.assertTrue(result['passed']);self.assertEqual(result['frames'],6)
    def test_shifted_timestamp_refuses(self):
        data=copy.deepcopy(self.data);data[1]['timestamp']+=1.;self.save(data)
        with self.assertRaisesRegex(ContentMismatch,'timestamp'):validate_metadata(self.run)
    def test_swapped_task_assignment_refuses(self):
        data=copy.deepcopy(self.data);data[1]['task_index']=1;self.save(data)
        with self.assertRaisesRegex(ContentMismatch,'task_index'):validate_metadata(self.run)
    def test_frame_episode_and_global_assignment_refuse(self):
        for field in ('frame_index','episode_index','index'):
            data=copy.deepcopy(self.data);data[1][field]+=1;self.save(data)
            with self.subTest(field=field),self.assertRaisesRegex(ContentMismatch,field):validate_metadata(self.run)
    def test_reordered_rows_refuse(self):
        data=copy.deepcopy(self.data);data[0],data[1]=data[1],data[0];self.save(data)
        with self.assertRaises(ContentMismatch):validate_metadata(self.run)

    def videos(self,order=None):
        import av
        def encode(path,values):
            path.parent.mkdir(parents=True,exist_ok=True)
            with av.open(str(path),'w',format='mov') as output:
                stream=output.add_stream('png',rate=30);stream.width=640;stream.height=480;stream.pix_fmt='rgb24';stream.codec_context.thread_count=1
                for value in values:
                    frame=av.VideoFrame.from_ndarray(np.full((480,640,3),value,dtype=np.uint8),format='rgb24')
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
        values=[20,90,210,40,120,240]
        for camera in ('top','wrist'):
            for i in range(2):encode(self.run/f'attempt-{i:04}'/(camera+'.mp4'),values[i*3:i*3+3])
            encode(self.root/'videos'/('observation.images.'+camera)/'chunk-000/file-000.mp4',[values[j] for j in (order or range(6))])
        info=json.loads((self.root/'meta/info.json').read_text());info['video_path']='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
        (self.root/'meta/info.json').write_text(json.dumps(info))
        path=self.root/'meta/episodes/file-000.parquet';rows=pq.read_table(path).to_pylist()
        for i,row in enumerate(rows):
            for camera in ('top','wrist'):
                key='videos/observation.images.'+camera+'/'
                row.update({key+'chunk_index':0,key+'file_index':0,key+'from_timestamp':i*.1,key+'to_timestamp':(i+1)*.1})
        pq.write_table(pa.Table.from_pylist(rows),path)

    def test_every_decoded_lossless_frame_and_pts_pass(self):
        self.videos();result=validate_videos(self.run,validate_metadata(self.run),sample_count=None)
        self.assertTrue(result['passed']);self.assertEqual(result['decoded_frame_pairs'],12)
        self.assertTrue(all(c['max_abs_rgb_error']==0 for e in result['episodes'] for c in e['cameras'].values()))

    def test_swapped_or_misordered_decoded_frames_refuse(self):
        for order in ([1,0,2,3,4,5],[3,4,5,0,1,2]):
            self.videos(order)
            with self.subTest(order=order),self.assertRaisesRegex(ContentMismatch,'video content'):
                validate_videos(self.run,validate_metadata(self.run),sample_count=None)

    def test_missing_or_extra_decoded_frame_count_refuses(self):
        self.videos([0,1,2,3,4])
        with self.assertRaisesRegex(ContentMismatch,'count'):validate_videos(self.run,validate_metadata(self.run),sample_count=None)

    def test_lossy_limits_are_applied_per_frame(self):
        black=np.zeros((480,640,3),np.uint8)
        compare_rgb(black,black+4,False)
        with self.assertRaisesRegex(ContentMismatch,'video content'):compare_rgb(black,black+255,False)
        with self.assertRaisesRegex(ContentMismatch,'video content'):compare_rgb(black,black+4,True)


if __name__=='__main__':unittest.main()
