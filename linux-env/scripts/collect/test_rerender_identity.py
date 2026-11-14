"""Held re-render selection with per-packet identity and cancellation boundaries."""
import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from frame_join import SensorAmbiguity,publisher_identity
from test_sensor_join import sensors,feed,message,publisher_info
from test_frame_join import image


class RerenderIdentityTests(unittest.TestCase):
    def test_retained_bootstrap_stamp_is_discarded_before_identity_checks(self):
        evidence=json.loads(Path(__file__).with_name('fixtures').joinpath('held-rerender-boundary.json').read_text())
        self.assertNotEqual(*evidence['candidate_sha256'])
        s=sensors();s.generation.install(dict(epoch=2,invalid_through_ns=evidence['invalid_through_ns']))
        for value in (11,22):s.put('wrist',image(evidence['stamp_ns'],value),None)
        self.assertIsNone(s.failure);self.assertEqual(s.stale_camera_messages['wrist'],2);self.assertEqual(s.camera_publishers,{})
        stamp=evidence['invalid_through_ns']+33333333
        s.put('wrist',image(stamp,11),publisher_info());s.put('wrist',image(stamp,22),publisher_info())
        self.assertEqual(s.rerender_messages['wrist'],1);self.assertIsNone(s.failure)

    def test_first_pixels_and_packet_metadata_survive_before_and_after_consumption(self):
        s=sensors();stamp=10**9;first=image(stamp,11)
        s.put('wrist',first,publisher_info());s.put('wrist',image(stamp,22),publisher_info())
        s.put('top',image(stamp,33),publisher_info(2));s.put('joint',message(stamp));s.put('tick',NS(data=json.dumps(dict(epoch=1,sim_ns=stamp))))
        _,entry=s.read(timeout=0);selected=copy.deepcopy(entry['wrist'][2])
        s.put('wrist',image(stamp,44),publisher_info())
        self.assertEqual(entry['wrist'][0].data[0],11);self.assertEqual(s.consumed[stamp]['wrist'],selected)
        self.assertEqual(s.rerender_messages['wrist'],2)
        event=next(iter(s.rerender_log.values()));self.assertEqual(event['selected_content_sha256'],selected['content_sha256'])
        self.assertNotEqual(event['last_content_sha256'],selected['content_sha256'])

    def test_distinct_publishers_are_terminal_even_when_pixels_are_identical(self):
        for reverse in (False,True):
            s=sensors();ids=(3,1) if reverse else (1,3)
            s.put('wrist',image(10**9,11),publisher_info(ids[0]))
            with self.assertRaises(SensorAmbiguity):s.put('wrist',image(10**9,11),publisher_info(ids[1]))
            candidates=s.failure.evidence['candidates']
            self.assertEqual(candidates[0]['content_sha256'],candidates[1]['content_sha256'])
            self.assertNotEqual(candidates[0]['publisher']['gid'],candidates[1]['publisher']['gid'])

    def test_nearby_stamps_or_layout_changes_are_not_rerenders(self):
        for kind in ('nearby_stamp','layout'):
            s=sensors();s.put('wrist',image(10**9,11),publisher_info());other=image(10**9,22)
            if kind=='nearby_stamp':other.header.stamp.nanosec=500
            else:other.encoding='rgba8'
            with self.subTest(kind=kind),self.assertRaises(SensorAmbiguity):s.put('wrist',other,publisher_info())

    def test_missing_packet_identity_is_terminal_but_bootstrap_is_not_bound(self):
        for info in (None,dict(publisher_gid=[0]*24,implementation_identifier='rmw_cyclonedds_cpp')):
            s=sensors()
            with self.assertRaises(SensorAmbiguity):s.put('wrist',image(10**9,11),info)
        s=sensors();s.generation.epoch=None
        s.put('wrist',image(10**9,11),None);self.assertIsNone(s.failure);self.assertEqual(s.camera_publishers,{})

    def test_native_transport_error_is_retained_for_callback_and_capture_containment(self):
        s=sensors();original=RuntimeError('native serialized take failed')
        with self.assertRaises(RuntimeError) as error:s.put('wrist',original,None)
        self.assertIs(error.exception,original)
        with self.assertRaises(RuntimeError) as error:s.read(timeout=0)
        self.assertIs(error.exception,original)

    def test_graph_boundary_guard_is_additional_and_never_equates_packet_ids(self):
        import test_review_regressions as probes
        c,r,entry=probes.ReviewRegressions().collector()
        endpoint=NS(endpoint_gid=[9]+[0]*23,node_name='camera_helper',node_namespace='/',topic_type='sensor_msgs/msg/Image',qos_profile=NS(depth=10,reliability=1,durability=2))
        c.node.get_publishers_info_by_topic=lambda topic:[endpoint]
        result=c.camera_graph();self.assertTrue(result['passed'])
        self.assertTrue(result['endpoints']['wrist']['endpoint_gid'].startswith('09'))
        c.node.get_publishers_info_by_topic=lambda topic:[endpoint,endpoint]
        with self.assertRaisesRegex(RuntimeError,'exactly one publisher'):c.camera_graph()
