"""Several slow but valid native queries must not trip an aggregate deadline."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from dart import TargetCandidate
from command_geometry import final_command,CommandCollision


class QueryBudgetTests(unittest.TestCase):
    def run_queries(self,latency,max_queries=None,verdict=True):
        clock=[0.];calls=[]
        def check(q):clock[0]+=latency;calls.append(q);return verdict,[]
        candidate=TargetCandidate(1,10**9,(.05,0.,0.,0.,0.,0.),(0.,)*5,(0.,)*6,(0.,)*6,(.05,0.,0.,0.,0.,0.))
        with patch('command_geometry.time',NS(monotonic=lambda:clock[0])):
            result=final_command(candidate,[0.]*6,[.031,0.,0.,0.,0.,0.],check,[(-3.,3.)]*6,max_queries=max_queries)
        return result,calls

    def test_nine_650ms_valid_answers_pass_despite_5850ms_total(self):
        result,calls=self.run_queries(.65)
        self.assertTrue(result['passed']);self.assertEqual(len(calls),9)
        self.assertAlmostEqual(result['query_timing']['total_query_wall_s'],5.85)
        self.assertAlmostEqual(result['query_timing']['max_query_wall_s'],.65)

    def test_missing_late_invalid_or_excessive_queries_still_refuse(self):
        with self.assertRaises(TimeoutError):self.run_queries(1.2)
        with self.assertRaises(CommandCollision):self.run_queries(.65,verdict=False)
        with self.assertRaisesRegex(RuntimeError,'query-count'):self.run_queries(.65,max_queries=8)
