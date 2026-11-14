import unittest
from run import fixture_pools,next_fixture


class FixtureTests(unittest.TestCase):
    def test_retry_never_counts_an_already_accepted_seed_twice(self):
        first=dict(params=dict(seed=330000));second=dict(params=dict(seed=330002))
        pool=[first,second]
        self.assertIs(next_fixture(pool,0,set()),first)
        self.assertIs(next_fixture(pool,1,{330000}),second)
        self.assertIs(next_fixture(pool,2,{330000}),second)
        with self.assertRaises(ValueError):next_fixture(pool,3,{330000,330002})

    def test_recovery_kinds_keep_their_ros_seed_fixtures(self):
        normal=dict(params=dict(seed=330000,recovery='none'))
        regrasp=dict(params=dict(seed=340000,recovery='regrasp'))
        pools=fixture_pools([normal,regrasp],['normal','regrasp'],1)
        self.assertEqual(pools,dict(normal=[normal],regrasp=[regrasp]))
        with self.assertRaises(ValueError):fixture_pools([normal,normal],['normal','regrasp'],1)
        with self.assertRaises(ValueError):fixture_pools([normal,normal],['normal'],2)


if __name__=='__main__':unittest.main()
