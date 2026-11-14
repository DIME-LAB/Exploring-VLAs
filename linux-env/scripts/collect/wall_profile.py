"""Same-host monotonic wall timing; diagnostic only, never a simulation clock."""
import math

PROFILE_POLICY=dict(version=1,clock='CLOCK_MONOTONIC on the same host',
    kit_boundary='advance requested to Kit post-update; proxy for render completion, not a GPU fence',
    camera_boundary='both RGB callbacks received; includes DDS transport and native publication ordering',
    native_state='in-process PhysX/body read; no per-tick bridge/run_python RPC',
    async_write='encoder pipe + row write on encoder worker; may overlap next tick',
    shares='mean stage / mean tick total; stages can overlap, shares are not an additive partition')


def milliseconds(start,end):return max(0.,end-start)/1e6


def profile_tick(snapshot,entry,read_start,read_end):
    native=snapshot.get('wall_profile_native',{})
    cameras=max(entry[k][1] for k in ('wrist','top'))
    step_start=native.get('step_start_ns');step_end=native.get('step_end_ns')
    result=dict(epoch=snapshot['epoch'],sim_ns=snapshot['sim_ns'],capture_start_ns=read_end,
        read_wait_ms=milliseconds(read_start,read_end),camera_receive_ns=cameras,
        kit_step_render_ms=milliseconds(step_start,step_end) if step_start is not None and step_end is not None else None,
        step_to_both_cameras_ms=milliseconds(step_start,cameras) if step_start is not None else None,
        image_transport_join_ms=milliseconds(step_end,read_end) if step_end is not None else None,
        native_state_read_ms=native.get('state_read_ms'),native_state_bridge_calls=0,
        native_step_start_ns=step_start,validity_queries=0,validity_query_ms=0.,
        expert_dart_compute_ms=0.,command_publish_ack_ms=0.,contract_validation_ms=0.,
        image_decode_ms=0.,row_queue_ms=0.,row_video_write_ms=0.,capture_preparation_ms=0.)
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
