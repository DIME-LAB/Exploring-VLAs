import unittest
import numpy as np
from probe import view_pose_errors,arguments


class PosePhaseTests(unittest.TestCase):
    def test_drawable_matrix_layouts_match_the_retained_pose(self):
        before=np.eye(4);before[:3,3]=[.3,.2,.8]
        after=before.copy();after[0,3]+=.02
        view=np.linalg.inv(before)
        for matrix in (view,view.T):
            result=view_pose_errors(matrix.reshape(-1).tolist(),before,after)
            self.assertLess(result['before_error'],1e-10)
            self.assertGreater(result['after_error'],.01)

    def test_a_view_of_the_arrival_pose_exposes_a_wrong_pixel_phase(self):
        before=np.eye(4);after=before.copy();after[0,3]=.02
        result=view_pose_errors(np.linalg.inv(after).reshape(-1).tolist(),before,after)
        self.assertGreater(result['before_error'],.01)
        self.assertLess(result['after_error'],1e-10)

    def test_invalid_view_metadata_is_refused(self):
        with self.assertRaises(ValueError):view_pose_errors([0.]*15,np.eye(4),np.eye(4))
        with self.assertRaises(ValueError):view_pose_errors([float('nan')]*16,np.eye(4),np.eye(4))


if __name__=='__main__':unittest.main()
