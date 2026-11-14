"""Primitive-gate live-pose fixtures and retained diagnostic media."""
import copy
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from cup_geometry import PROFILE,WALLS,wall_checks
from success import contained,settled_containment
from test_tick_contract import checked_window
from diagnostics import final_stills, png_dimensions
import test_safety


class CupProfileTests(unittest.TestCase):
    def fixture(self):
        return json.loads(Path(__file__).with_name('fixtures').joinpath('live-cup-rest-310000.json').read_text())

    def test_retained_live_pose_is_inside_the_primitive_planes(self):
        data=self.fixture()
        result=settled_containment(checked_window(data['snapshots']),data['object'],data['color'])
        self.assertTrue(result['passed'])
        check=result['checks'][-1]
        self.assertAlmostEqual(check['max_corner_radius_m'],.02524526772523616,delta=1e-12)
        self.assertGreater(check['min_wall_clearance_m'],.0005)
        self.assertEqual(len(check['wall_checks']),96)

    def test_same_live_pose_cannot_pass_on_rim_outside_or_with_protruding_corner(self):
        data=self.fixture();sample=data['snapshots'][-1];name=data['object'];cup=sample['bodies']['cup_red']
        for kind in ('rim','outside','protruding','below_floor','moving','cup_tilt'):
            s=copy.deepcopy(sample);body=s['bodies'][name]
            if kind=='rim':body['position'][2]=cup['position'][2]+.09652+.0064
            elif kind=='outside':body['position'][0]+= .08
            elif kind=='protruding':body['position'][0]+= .02
            elif kind=='below_floor':body['position'][2]=cup['position'][2]+.0064+.0005
            elif kind=='moving':body['velocity'][0]=.02
            else:s['bodies']['cup_red']['orientation']=[math.cos(.1),math.sin(.1),0,0]
            with self.subTest(kind=kind):self.assertFalse(contained(s,name,'red')['passed'])

    def test_height_changes_radial_admission_without_accepting_rim_or_floor_failures(self):
        low=test_safety.SafetyTests().snapshot(.012);low['bodies']['red_2x3_0']['position'][0]=.012
        high=copy.deepcopy(low);high['bodies']['red_2x3_0']['position'][2]=.07
        self.assertFalse(contained(low,'red_2x3_0','red')['passed'])
        self.assertTrue(contained(high,'red_2x3_0','red')['passed'])
        n=WALLS[0]['interior_plane']['normal'];offset=WALLS[0]['interior_plane']['offset_m']
        self.assertLess(offset-n[2]*.003,offset-n[2]*.08)

    def test_every_intermediate_box_point_is_inside_the_linear_halfplanes(self):
        from pose_math import rotate
        data=self.fixture();sample=data['snapshots'][-1];name=data['object'];body=sample['bodies'][name];cup=sample['bodies']['cup_'+data['color']]
        inv=[cup['orientation'][0]]+[-v for v in cup['orientation'][1:]]
        corners=[rotate(inv,[a+b-c for a,b,c in zip(body['position'],rotate(body['orientation'],local),cup['position'])]) for local in sample['local_corners'][name]]
        for a in corners:
            for b in corners:
                for t in (0.,.1,.5,.9,1.):
                    point=[(1-t)*x+t*y for x,y in zip(a,b)]
                    self.assertTrue(all(c['passed'] for c in wall_checks([point])))


class DiagnosticMediaTests(unittest.TestCase):
    def test_final_stills_select_exact_last_native_frame_without_extra_steps(self):
        calls=[]
        with patch('pathlib.Path.unlink'):
            result=final_stills(Path('/confined/attempt'),[dict(sim_ns=33),dict(sim_ns=66)],
                run=lambda command,**kw:calls.append((command,kw)),verify=lambda path:[640,480])
        self.assertEqual(result['frame_index'],1);self.assertEqual(result['sim_ns'],66)
        self.assertTrue(result['saved']);self.assertEqual(len(calls),2)
        for command,kw in calls:
            self.assertIn('select=eq(n\\,1)',command);self.assertEqual(kw['timeout'],20);self.assertTrue(kw['check'])
        self.assertIn('final-top.png',calls[1][0][-1])

    def test_no_still_is_invented_when_no_frame_was_recorded(self):
        def unexpected(*args,**kwargs):raise AssertionError('No encoder call expected')
        self.assertFalse(final_stills(Path('/confined'),[],unexpected)['saved'])

    def test_png_must_exist_be_nonempty_and_decode(self):
        from types import SimpleNamespace as NS
        from PIL import Image
        buffer=io.BytesIO();Image.new('RGB',(32,24),(70,80,90)).save(buffer,format='PNG')
        self.assertEqual(png_dimensions(NS(read_bytes=lambda:buffer.getvalue())),[32,24])
        for raw in (b'',b'not a PNG',buffer.getvalue()[:40]):
            with self.subTest(raw_bytes=len(raw)):
                with self.assertRaises((ValueError,OSError)):
                    png_dimensions(NS(read_bytes=lambda:raw))
        def missing():raise FileNotFoundError('No PNG extracted')
        with self.assertRaises(FileNotFoundError):png_dimensions(NS(read_bytes=missing))

    def test_successful_ffmpeg_exit_without_an_image_cannot_report_saved(self):
        calls=[]
        def missing(path):raise FileNotFoundError(str(path))
        with patch('pathlib.Path.unlink') as removed:
            with self.assertRaises(FileNotFoundError):
                final_stills(Path('/confined'),[dict(sim_ns=1)],
                    run=lambda *a,**kw:calls.append(a),verify=missing)
        self.assertEqual(len(calls),1);self.assertEqual(removed.call_count,2)

if __name__=='__main__':unittest.main()
