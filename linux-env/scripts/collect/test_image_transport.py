"""Native buffer ownership and isolated executor forwarding, with no ROS runtime."""
import asyncio,ctypes as C,unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from image_transport import ImageTake,publisher_executor_type


class ImageTransportTests(unittest.TestCase):
    def reader(self,result=1):
        reader=object.__new__(ImageTake);reader.gid_size=24;buffer=C.create_string_buffer(b'abcd');freed=Mock();events=[]
        def take(pointer,payload,length,gid,implementation,implementation_size,source,received,publication,reception,error,error_size):
            events.append('native_take');self.assertEqual(pointer,123)
            if result<0:error.value=b'original RCL failure';return result
            if not result:return 0
            payload._obj.value=C.addressof(buffer);length._obj.value=4;gid[0]=7;implementation.value=b'rmw_cyclonedds_cpp'
            source._obj.value=10;received._obj.value=11;publication._obj.value=12;reception._obj.value=13;return 1
        class Handle:
            pointer=123
            def __enter__(self):events.append('borrow');return self
            def __exit__(self,*args):events.append('return_borrow')
        reader.lib=NS(collection_take=take,collection_free=freed)
        sub=NS(handle=Handle(),msg_type='image')
        return reader,sub,freed,buffer,events

    def test_payload_is_released_once_after_deserialization_without_owning_subscription(self):
        reader,sub,freed,buffer,events=self.reader();decoded=object()
        with patch.dict('sys.modules',{'rclpy.serialization':NS(deserialize_message=Mock(return_value=decoded))}):message,info=reader.take(sub)
        self.assertIs(message,decoded);self.assertEqual(info['publisher_gid'],[7]+[0]*23)
        self.assertEqual(info['publication_sequence_number'],12);self.assertEqual(events,['borrow','native_take','return_borrow'])
        freed.assert_called_once();self.assertEqual(freed.call_args.args[0].value,C.addressof(buffer))

    def test_deserialize_error_still_releases_native_copy(self):
        reader,sub,freed,buffer,events=self.reader()
        with patch.dict('sys.modules',{'rclpy.serialization':NS(deserialize_message=Mock(side_effect=ValueError('bad CDR')))}):
            with self.assertRaisesRegex(ValueError,'bad CDR'):reader.take(sub)
        freed.assert_called_once()

    def test_empty_take_and_rcl_errors_never_free_an_unowned_payload(self):
        for result in (0,-1):
            reader,sub,freed,buffer,events=self.reader(result)
            with patch.dict('sys.modules',{'rclpy.serialization':NS(deserialize_message=Mock())}):
                if not result:self.assertIsNone(reader.take(sub))
                else:
                    with self.assertRaisesRegex(RuntimeError,'original RCL failure'):reader.take(sub)
            freed.assert_not_called()

    def test_executor_preserves_regular_subscriptions_and_forwards_info_or_original_error(self):
        calls=[]
        class Base:
            def _take_subscription(self,sub):return 'regular take'
            async def _execute_subscription(self,sub,message):calls.append(('regular',message))
        async def invoke(callback,*args):return callback(*args)
        with patch.dict('sys.modules',{'rclpy.executors':NS(MultiThreadedExecutor=Base,await_or_execute=invoke)}):cls=publisher_executor_type()
        executor=object.__new__(cls);packet=object();info=dict(publisher_gid=[7]+[0]*23)
        executor.image_take=NS(take=Mock(return_value=(packet,info)))
        normal=NS();marked=NS(collection_publisher_info=True,callback=lambda *args:calls.append(args))
        self.assertEqual(executor._take_subscription(normal),'regular take')
        asyncio.run(executor._execute_subscription(normal,'message'));self.assertEqual(calls.pop(),('regular','message'))
        value=executor._take_subscription(marked);asyncio.run(executor._execute_subscription(marked,value));self.assertEqual(calls.pop(),(packet,info))
        original=RuntimeError('native transport');executor.image_take.take.side_effect=original
        value=executor._take_subscription(marked);asyncio.run(executor._execute_subscription(marked,value));self.assertEqual(calls.pop(),(original,None))
