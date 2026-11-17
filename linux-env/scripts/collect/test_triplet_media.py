"""Generic tolerance-valid groups still need the stricter production gate."""
import unittest
import numpy as np
from triplet_media import measure


class TripletMediaTests(unittest.TestCase):
    def fixture(self):
        hashes=['a'*64]*3;states=np.zeros((3,6));actions=np.zeros((3,50,6))
        for i in range(3):actions[i,:,0]=i*.01
        images={camera:[np.full((480,640,3),100+i,np.uint8) for i in range(3)] for camera in ('top','wrist')}
        return hashes,states,actions,images
    def test_same_source_first_state_and_each_camera_pass_with_distinct_labels(self):
        result=measure(*self.fixture());self.assertTrue(result['passed']);self.assertEqual(len(result['image_pairs']),6)
        self.assertTrue(all(p['first_different_frame']==0 for p in result['label_pairs']))
    def test_generic_three_hash_or_submicrometre_state_difference_refuses_production(self):
        for kind in ('hash','state','signed_zero'):
            hashes,states,actions,images=self.fixture()
            if kind=='hash':hashes[1]='b'*64
            elif kind=='state':states[1,0]=1e-8
            else:states[1,0]=-0.
            with self.subTest(kind=kind):self.assertIn('exact_identity',measure(hashes,states,actions,images)['reasons'])
    def test_one_bad_camera_cannot_hide_behind_the_other(self):
        for camera in ('top','wrist'):
            hashes,states,actions,images=self.fixture();images[camera][1].fill(160)
            with self.subTest(camera=camera):self.assertIn('image_psnr',measure(hashes,states,actions,images)['reasons'])
    def test_frame0_hold_is_allowed_when_first50_departure_labels_differ(self):
        hashes,states,actions,images=self.fixture();actions[:,0,:]=0
        result=measure(hashes,states,actions,images);self.assertTrue(result['passed'])
        self.assertTrue(all(p['first_different_frame']==1 for p in result['label_pairs']))
        actions[:]=0;self.assertIn('departure_labels',measure(hashes,states,actions,images)['reasons'])
    def test_incomplete_state_label_or_camera_refuses(self):
        for kind in ('nan','short','shape','camera'):
            hashes,states,actions,images=self.fixture()
            if kind=='nan':actions[0,0,0]=np.nan
            elif kind=='short':actions=actions[:,:49]
            elif kind=='shape':images['wrist'][1]=images['wrist'][1][:20]
            else:images.pop('top')
            with self.subTest(kind=kind),self.assertRaises(ValueError):measure(hashes,states,actions,images)


if __name__=='__main__':unittest.main()
