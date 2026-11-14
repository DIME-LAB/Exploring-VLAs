import unittest
from run import fixture_pools


class FixtureTests(unittest.TestCase):
    def test_recovery_kinds_keep_their_ros_seed_fixtures(self):
        normal=dict(params=dict(seed=330000,recovery='none'))
        regrasp=dict(params=dict(seed=340000,recovery='regrasp'))
        pools=fixture_pools([normal,regrasp],['normal','regrasp'],1)
        self.assertEqual(pools,dict(normal=[normal],regrasp=[regrasp]))
        with self.assertRaises(ValueError):fixture_pools([normal,normal],['normal','regrasp'],1)
        with self.assertRaises(ValueError):fixture_pools([normal,normal],['normal'],2)


if __name__=='__main__':unittest.main()
