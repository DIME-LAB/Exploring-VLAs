"""Retained live 1 ns re-renders select first pixels for one physical tick."""
import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from frame_join import SensorAmbiguity
from test_sensor_join import sensors,message,publisher_info
from test_frame_join import image

FIXTURE=json.loads(Path(__file__).with_name('fixtures').joinpath('native-tick-rerender.json').read_text())


def native(s,stamp,epoch=8):s.put('tick',NS(data=json.dumps(dict(epoch=epoch,sim_ns=stamp))))


def setup():
    s=sensors();tick=FIXTURE['native_tick']['sim_ns'];s.generation.install(dict(epoch=8,invalid_through_ns=tick-33333333))
    return s,tick


class NativeCameraTickTests(unittest.TestCase):
    def test_live_pair_keeps_first_raw_stamp_and_pixels_in_every_arrival_order(self):
        first,last=[f['stamp_ns'] for f in FIXTURE['sensor_evidence']['candidates']]
        self.assertEqual(last-first,1)
        self.assertNotEqual(*[f['content_sha256'] for f in FIXTURE['sensor_evidence']['candidates']])
        for order in ('native_first','camera_first','after_consumed'):
            with self.subTest(order=order):
                s,tick=setup()
                if order=='native_first':native(s,tick)
                s.put('wrist',image(first,11),publisher_info())
                if order!='after_consumed':s.put('wrist',image(last,22),publisher_info())
                if order!='native_first':native(s,tick)
                s.put('joint',message(tick));s.put('top',image(tick,33),publisher_info(2))
                stamp,entry=s.read(timeout=0);selected=copy.deepcopy(entry['wrist'][2])
                if order=='after_consumed':s.put('wrist',image(last,22),publisher_info())
                self.assertEqual(stamp,first);self.assertEqual(entry['wrist'][0].data[0],11)
                self.assertEqual(selected['native_tick'],dict(epoch=8,sim_ns=tick))
                self.assertEqual(entry['wrist'][2],selected);self.assertEqual(s.consumed[first]['wrist'],selected)
                self.assertEqual(s.rerender_messages['wrist'],1);self.assertIsNone(s.failure)

    def test_one_native_tick_allows_the_full_independent_producer_bound(self):
        s,tick=setup();s.put('wrist',image(tick-1000,11),publisher_info());s.put('wrist',image(tick+1000,22),publisher_info())
        native(s,tick);s.put('joint',message(tick));s.put('top',image(tick,33),publisher_info(2))
        _,entry=s.read(timeout=0);self.assertEqual(entry['wrist'][0].data[0],11)
        self.assertEqual(entry['wrist'][2]['native_tick']['sim_ns'],tick)

    def test_pixels_are_provisional_until_native_tick_arrives(self):
        s,tick=setup();s.put('wrist',image(tick-2,11),publisher_info());s.put('top',image(tick,33),publisher_info(2));s.put('joint',message(tick))
        with self.assertRaises(TimeoutError):s.read(timeout=0)
        self.assertEqual(s.consumed,{})
        native(s,tick);self.assertEqual(s.read(timeout=0)[1]['wrist'][0].data[0],11)

    def test_multiple_native_ticks_stay_terminal_even_with_identical_pixels(self):
        for after in (False,True):
            with self.subTest(after=after):
                s,tick=setup();native(s,tick);s.put('wrist',image(tick-2,11),publisher_info())
                if after:
                    s.put('joint',message(tick));s.put('top',image(tick,33),publisher_info(2));s.read(timeout=0)
                with self.assertRaises(SensorAmbiguity) as error:native(s,tick+500)
                self.assertEqual(len(error.exception.evidence['native_ticks']),2)
                with self.assertRaises(SensorAmbiguity):s.read(timeout=0)

    def test_missing_common_native_or_changed_publisher_layout_stays_terminal(self):
        for kind in ('offset','publisher','layout'):
            with self.subTest(kind=kind):
                s,tick=setup();native(s,tick);s.put('wrist',image(tick,11),publisher_info())
                other=image(tick+1500 if kind=='offset' else tick,22)
                if kind=='layout':other.encoding='rgba8'
                with self.assertRaises(SensorAmbiguity):s.put('wrist',other,publisher_info(9 if kind=='publisher' else 1))

    def test_export_contract_requires_native_ids_and_copied_identity_stays_exact(self):
        from tick_contract import camera_tick_errors
        from test_tick_contract import checked_window
        row=checked_window([{}])[0]['tick_contract']['row']
        self.assertTrue(camera_tick_errors(row,required=True))
        row['camera_identity']={key:dict(stamp_ns=row[key+'_stamp_ns'],native_tick=dict(epoch=row['epoch'],sim_ns=row['pose_stamp_ns'])) for key in ('wrist','top')}
        self.assertEqual(camera_tick_errors(row,required=True),[])
        changed=copy.deepcopy(row);changed['camera_identity']['wrist']['native_tick']['sim_ns']+=1
        self.assertTrue(any('"rule": "copy"' in error for error in camera_tick_errors(changed,required=True)))
        changed=copy.deepcopy(row);changed['camera_identity']['wrist']['stamp_ns']-=1
        self.assertTrue(any('"rule": "copy"' in error for error in camera_tick_errors(changed,required=True)))

    def test_second_native_tick_at_opposite_header_bound_is_immediately_terminal(self):
        for direction in (-1,1):
            with self.subTest(direction=direction):
                s,tick=setup();native(s,tick)
                first=tick-direction*1000;other=tick+direction*1000
                s.put('wrist',image(first,11),publisher_info());s.put('wrist',image(other,22),publisher_info())
                s.put('top',image(tick,33),publisher_info(2));s.put('joint',message(tick));s.read(timeout=0)
                # A second native tick can be3us from the selected header and
                # still be within1us of a retained alternate. It is ambiguous.
                with self.assertRaises(SensorAmbiguity) as error:native(s,tick+direction*2000)
                self.assertEqual(len(error.exception.evidence['native_ticks']),2)
