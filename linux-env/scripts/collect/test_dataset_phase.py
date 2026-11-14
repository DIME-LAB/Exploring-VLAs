"""Frame-level phase flags survive export and distinguish capped miss labels."""
import copy,unittest
from dataset_phase import observed_phase,phase_code,feature,PHASES,FEATURE

class DatasetPhaseTests(unittest.TestCase):
    def test_normal_prefix_and_corrective_flags_are_explicit(self):
        self.assertEqual(observed_phase('none',3,None),'normal')
        self.assertEqual(observed_phase('regrasp',3,None),'miss_prefix')
        self.assertEqual(observed_phase('regrasp',3,3),'corrective')
        self.assertEqual(observed_phase('regrasp',4,3),'miss_prefix')
        self.assertEqual(feature(),dict(dtype='int64',shape=(1,),names=['phase']))

    def test_export_checks_flag_against_native_reopen_boundary(self):
        receipt=dict(params=dict(recovery='regrasp'),recovery_report=dict(clearance_end_sim_ns=100))
        for stamp,phase in ((99,'miss_prefix'),(100,'miss_prefix'),(101,'corrective')):
            row=dict(epoch=1,pose_stamp_ns=stamp,sim_ns=stamp,state_source='isaac_articulation',state=[0.]*6,physical_bodies={},collection_phase=phase)
            self.assertEqual(phase_code(row,receipt,required=True),PHASES[phase])
            bad=dict(row,collection_phase='normal')
            with self.assertRaises(ValueError):phase_code(bad,receipt,required=True)
        with self.assertRaises(ValueError):phase_code(dict(row,collection_phase=None),receipt)

    def test_normal_legacy_export_is_compatible_but_new_missing_flags_refuse(self):
        receipt=dict(params=dict(recovery='none'))
        self.assertEqual(phase_code({},receipt),0)
        with self.assertRaises(ValueError):phase_code({},receipt,required=True)
        with self.assertRaises(ValueError):phase_code(dict(collection_phase='miss_prefix'),receipt)

    def test_scalar_and_length_one_hf_columns_round_trip_without_indexing_scalar(self):
        import numpy as np
        from types import SimpleNamespace as NS
        from dataset_phase import stored_phase_codes
        dataset=NS(hf_dataset={FEATURE:[np.asarray(0),np.asarray([1]),np.asarray(2)]})
        self.assertEqual(stored_phase_codes(dataset),[0,1,2])
        dataset.hf_dataset[FEATURE]=[np.asarray(1.5)]
        with self.assertRaises(ValueError):stored_phase_codes(dataset)
