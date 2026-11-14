"""Same-host monotonic wall timing; diagnostic only, never a simulation clock."""
import math,copy,json
from comparisons import comparison

PROFILE_POLICY=dict(version=3,clock='CLOCK_MONOTONIC on the same host',
    kit_boundary='advance requested to Kit post-update; proxy for render completion, not a GPU fence',
    camera_boundary='both RGB callbacks received; includes DDS transport and native publication ordering',
    native_state='in-process PhysX/body read; no per-tick bridge/run_python RPC',
    async_write='write-profiles.jsonl links immutable capture rows by profile_id; queue includes request to dequeue; overlaps next tick',
    shares='mean stage / mean tick total; stages can overlap, shares are not an additive partition')


def milliseconds(start,end):return max(0.,end-start)/1e6


def profile_tick(snapshot,entry,read_start,read_end):
    native=snapshot.get('wall_profile_native',{})
    cameras=max(entry[k][1] for k in ('wrist','top'))
    step_start=native.get('step_start_ns');step_end=native.get('step_end_ns')
    result=dict(profile_id=str(snapshot['epoch'])+':'+str(snapshot['sim_ns'])+':'+str(read_start),epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],capture_start_ns=read_end,
        native_sim_ns=snapshot['sim_ns'],camera_stamp_ns=(entry['wrist'][2]['stamp_ns'] if len(entry['wrist'])>2 else None),
        read_wait_ms=milliseconds(read_start,read_end),camera_receive_ns=cameras,
        kit_step_render_ms=milliseconds(step_start,step_end) if step_start is not None and step_end is not None else None,
        step_to_both_cameras_ms=milliseconds(step_start,cameras) if step_start is not None else None,
        image_transport_join_ms=milliseconds(step_end,read_end) if step_end is not None else None,
        native_state_read_ms=native.get('state_read_ms'),native_state_bridge_calls=0,
        native_step_start_ns=step_start,validity_queries=0,validity_query_ms=0.,
        expert_dart_compute_ms=0.,command_publish_ack_ms=0.,contract_validation_ms=0.,
        image_decode_ms=0.,row_queue_ms=None,row_video_write_ms=None,capture_preparation_ms=0.,recorded=False)
    decode=[entry[key][2].get('publisher',{}).get('transport_take_decode_ms') if len(entry[key])>2 else None for key in ('wrist','top')]
    result['native_image_take_decode_ms']=sum(decode) if all(isinstance(v,(int,float)) for v in decode) else None
    return result


def summary(rows):
    fields=sorted({key for row in rows for key,value in row.items() if key.endswith('_ms') and isinstance(value,(int,float))})
    stats={}
    for key in fields:
        values=sorted(float(row[key]) for row in rows if isinstance(row.get(key),(int,float)) and math.isfinite(row[key]))
        if values:stats[key]=dict(count=len(values),mean=sum(values)/len(values),p95=values[max(0,math.ceil(.95*len(values))-1)])
    total=stats.get('tick_total_ms',{}).get('mean')
    for key,value in stats.items():value['mean_share_of_tick']=value['mean']/total if total else None
    queries=[row.get('validity_queries',0) for row in rows]
    return dict(policy=PROFILE_POLICY,ticks=len(rows),stages=stats,validity_query_count=sum(queries),
                mean_validity_queries=sum(queries)/len(queries) if queries else 0.)


def joined_profiles(captures,writes):
    linked={row['profile_id']:row for row in writes}
    if len(linked)!=len(writes):raise ProfileContractError([dict(field='profile_id',rule='copy',error='Duplicate async write profile identity')])
    result=[]
    for original in captures:
        row=copy.deepcopy(original);write=linked.pop(row['profile_id'],None)
        if write is not None:
            native=row.get('native_sim_ns',row['sim_ns'])
            camera=row.get('camera_stamp_ns',row['sim_ns'])
            proofs=[comparison('async profile epoch',write['epoch'],row['epoch']),
                comparison('async native tick',write.get('native_sim_ns'),native),
                comparison('async native alias',write['sim_ns'],native),
                comparison('async selected camera stamp',write.get('camera_stamp_ns'),camera),
                comparison('native/camera producer agreement',native,camera,'stamp_ns')]
            failed=[proof for proof in proofs if not proof['passed']]
            if failed:raise ProfileContractError(failed)
            row.update({key:write[key] for key in ('row_queue_ms','row_video_write_ms')})
        result.append(row)
    if linked:raise RuntimeError('Async write has no matching capture profile')
    return result


class ProfileContractError(RuntimeError):
    def __init__(self,proofs):
        self.proofs=proofs
        super().__init__('Async write profile contract failed: '+json.dumps(proofs,sort_keys=True))
