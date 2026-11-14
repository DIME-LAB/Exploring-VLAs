"""Read compute AND graphics process attribution without importing a CUDA runtime.

ABI: /usr/local/cuda/include/nvml.h nvmlProcessInfo_v2_t and the *_v3 queries.
The two APIs can report the same process memory; consumers must not sum it twice.
"""
import ctypes as C


class ProcessInfo(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),
              ('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]


def sample(cards,library=None):
    """Unsupported/permission/unknown memory stays unverified, never inferred zero."""
    rows=[];coverage={};errors=[];initialized=False
    try:
        lib=library if library is not None else C.CDLL('libnvidia-ml.so.1')
        def call(name,args,types):
            fn=getattr(lib,name);fn.argtypes=types;fn.restype=C.c_int;return fn(*args)
        result=call('nvmlInit_v2',[],[])
        if result:raise RuntimeError('nvmlInit_v2 returned '+str(result))
        initialized=True
        for card in cards:
            uuid=card['uuid'];coverage[uuid]={};handle=C.c_void_p()
            result=call('nvmlDeviceGetHandleByUUID',[uuid.encode(),C.byref(handle)],[C.c_char_p,C.POINTER(C.c_void_p)])
            if result:raise RuntimeError('NVML device lookup failed: '+str(result))
            for api,method in [('compute','nvmlDeviceGetComputeRunningProcesses_v3'),('graphics','nvmlDeviceGetGraphicsRunningProcesses_v3')]:
                try:
                    count=C.c_uint(0)
                    result=call(method,[handle,C.byref(count),None],[C.c_void_p,C.POINTER(C.c_uint),C.POINTER(ProcessInfo)])
                    if result not in (0,7):raise RuntimeError(method+' returned '+str(result))
                    for _ in range(4):
                        capacity=max(16,count.value+8);buffer=(ProcessInfo*capacity)();count.value=capacity
                        result=call(method,[handle,C.byref(count),buffer],[C.c_void_p,C.POINTER(C.c_uint),C.POINTER(ProcessInfo)])
                        if result!=7:break
                    if result:raise RuntimeError(method+' returned '+str(result))
                    if count.value>capacity:raise RuntimeError('NVML returned an invalid process count')
                    for item in buffer[:count.value]:
                        memory=None if item.usedGpuMemory==2**64-1 else int(item.usedGpuMemory)
                        rows.append(dict(pid=int(item.pid),gpu_uuid=uuid,memory_bytes=memory,api=api))
                    coverage[uuid][api]='verified'
                except Exception as exc:
                    coverage[uuid][api]='unverified';errors.append(dict(gpu_uuid=uuid,api=api,error=repr(exc)))
    except Exception as exc:errors.append(dict(error=repr(exc)))
    finally:
        if initialized:
            try:
                result=lib.nvmlShutdown()
                if result:errors.append(dict(error='nvmlShutdown returned '+str(result)))
            except Exception as exc:errors.append(dict(error=repr(exc)))
    return dict(rows=rows,coverage=coverage,errors=errors)
