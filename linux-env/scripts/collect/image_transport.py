"""Collector-only Humble adapter retaining the actual packet's RMW publisher GID.

Humble's executor discards message info and its pybind take lacks publisher_gid.
The local C ABI uses rcl_take_serialized_message on that SAME held subscription;
Python's normal deserialize_message still constructs the sensor_msgs/Image.
"""
import ctypes as C,fcntl,hashlib,os,subprocess,uuid,time,inspect
from pathlib import Path
from storage import confined,atomic_json

_LOADED_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_native():
    source=Path(__file__).with_name('image_take.c');ros=Path('/opt/ros/humble')
    headers=[ros/'include/rmw/rmw/types.h',ros/'include/rcl/rcl/subscription.h']
    raw=source.read_bytes()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=_LOADED_SHA256:raise RuntimeError('Image adapter source changed; restart collector')
    digest=hashlib.sha256(raw+Path(__file__).read_bytes()+b''.join(p.read_bytes() for p in headers)).hexdigest()
    root=confined(Path(os.environ['SOARM_INSTANCE_ROOT'])/'native-image-transport'/digest)
    root.mkdir(parents=True,exist_ok=True);library=root/'image_take.so'
    fd=os.open(root/'build.lock',os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        if not library.exists():
            temporary=root/('image_take.'+uuid.uuid4().hex+'.so')
            staged=root/'image_take.c';fd_source=os.open(staged,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd_source,'wb') as stream:stream.write(raw)
            command=['cc','-std=c11','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror',str(staged),'-o',str(temporary)]
            command += ['-I'+str(ros/'include'/name) for name in
                ('rcl','rmw','rcutils','rosidl_runtime_c','rosidl_typesupport_interface','rcl_yaml_param_parser','rcl_logging_interface')]
            command += ['-L'+str(ros/'lib'),'-lrcl','-lrmw','-lrcutils']
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=60)
                atomic_json(root/'build.json',dict(argv=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
                if result.returncode:raise RuntimeError('Native image transport build failed: '+result.stderr)
                os.replace(temporary,library)
            finally:
                if temporary.exists():temporary.unlink()
    finally:os.close(fd)
    return library,dict(source_sha256=hashlib.sha256(raw).hexdigest(),python_source_sha256=_LOADED_SHA256,abi_sha256=digest,
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),library=str(library),ros_prefix=str(ros))


class ImageTake:
    def __init__(self):
        path,self.metadata=build_native();self.lib=C.CDLL(str(path))
        self.lib.collection_gid_size.restype=C.c_size_t;self.gid_size=self.lib.collection_gid_size()
        if self.gid_size!=24:raise RuntimeError('Unexpected ROS publisher GID ABI: '+str(self.gid_size))
        self.lib.collection_free.argtypes=[C.c_void_p];self.lib.collection_free.restype=None
        self.lib.collection_take.argtypes=[C.c_size_t,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t),C.c_void_p,
            C.c_void_p,C.c_size_t,C.POINTER(C.c_int64),C.POINTER(C.c_int64),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.c_void_p,C.c_size_t]
        self.lib.collection_take.restype=C.c_int

    def take(self,subscription):
        from rclpy.serialization import deserialize_message
        start=time.monotonic_ns()
        payload=C.c_void_p();length=C.c_size_t();gid=(C.c_ubyte*self.gid_size)();implementation=C.create_string_buffer(128)
        source=C.c_int64();received=C.c_int64();publication=C.c_uint64();reception=C.c_uint64();error=C.create_string_buffer(1024)
        with subscription.handle:
            result=self.lib.collection_take(subscription.handle.pointer,C.byref(payload),C.byref(length),gid,
                implementation,len(implementation),C.byref(source),C.byref(received),C.byref(publication),C.byref(reception),error,len(error))
        if result<0:raise RuntimeError(error.value.decode(errors='replace'))
        if not result:return None
        try:
            serialized=C.string_at(payload,length.value)
            message=deserialize_message(serialized,subscription.msg_type)
        finally:self.lib.collection_free(payload)
        info=dict(transport_take_decode_ms=(time.monotonic_ns()-start)/1e6,publisher_gid=list(gid),implementation_identifier=implementation.value.decode(),
            source_timestamp=source.value,received_timestamp=received.value,
            publication_sequence_number=publication.value,reception_sequence_number=reception.value)
        return message,info


def publisher_executor_type():
    from rclpy.executors import MultiThreadedExecutor,await_or_execute
    if (not hasattr(MultiThreadedExecutor,'_take_subscription') or not hasattr(MultiThreadedExecutor,'_execute_subscription') or
            len(inspect.signature(MultiThreadedExecutor._take_subscription).parameters)!=2 or
            len(inspect.signature(MultiThreadedExecutor._execute_subscription).parameters)!=3 or
            not inspect.iscoroutinefunction(MultiThreadedExecutor._execute_subscription)):
        raise RuntimeError('Image metadata adapter requires the tested Humble executor take/execute API')
    class PublisherInfoExecutor(MultiThreadedExecutor):
        def __init__(self,**kwargs):
            self.image_take=ImageTake();super().__init__(**kwargs)
        def _take_subscription(self,subscription):
            if getattr(subscription,'collection_publisher_info',False):
                try:return self.image_take.take(subscription)
                except Exception as exc:return exc,None
            return super()._take_subscription(subscription)
        async def _execute_subscription(self,subscription,message):
            if getattr(subscription,'collection_publisher_info',False):
                if message is not None:await await_or_execute(subscription.callback,*message)
            else:await super()._execute_subscription(subscription,message)
    return PublisherInfoExecutor
