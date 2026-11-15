"""CPU regressions for the selected-clock action schedule and RTC delay."""
import math
import unittest
from sim_time import TickSchedule, PacingClock, delay_steps
from phase3 import manifest, command
from pathlib import Path

class TimingTests(unittest.TestCase):
    def test_identical_sim_time_at_different_rtfs(self):
        traces=[];walls=[]
        for rtf in (.3,1.):
            clock=TickSchedule(100.,30)
            stamps=[];wall=[]
            for n in range(301):
                sim=100+n/30
                expected,missed=clock.take(sim)
                self.assertEqual(missed,0)
                stamps.append((sim,expected));wall.append(n/30/rtf)
            traces.append(stamps);walls.append(wall)
        self.assertEqual(traces[0],traces[1])
        self.assertAlmostEqual(walls[0][-1]/walls[1][-1],1/.3)

    def test_clock_jump_skips_and_never_bursts(self):
        schedule=TickSchedule(0,30)
        self.assertEqual(schedule.take(0),(0,0))
        expected,missed=schedule.take(.2)
        self.assertEqual(missed,5);self.assertAlmostEqual(expected,.2)
        self.assertGreater(schedule.deadline,.2)

    def test_reset_and_missing_clock_fail_closed(self):
        clock=PacingClock.__new__(PacingClock)
        import threading,time
        clock.use_sim_time=True;clock.condition=threading.Condition();clock.value=None
        clock.error=None;clock.stale_wall_s=2;clock.received=time.monotonic()
        with self.assertRaises(TimeoutError):clock.now()
        clock.feed(10);self.assertEqual(clock.now(),10)
        clock.feed(9)
        with self.assertRaises(RuntimeError):clock.now()

    def test_delay_uses_sim_clock_not_wall_rate(self):
        for rtf in (.3,1):
            # Same 300ms wall prediction spans different simulated durations.
            self.assertEqual(delay_steps(.3*rtf,10,10+.1*rtf,30,40),math.ceil(.4*rtf*30-1e-8))
        self.assertEqual(delay_steps(100,1,2,30,7),7)
        self.assertEqual(delay_steps(0,10+1/30,10,30,40),0)
        with self.assertRaises(ValueError):delay_steps(.1,3,2,30,10)

    def test_wall_mode_needs_no_ros(self):
        a=PacingClock(False);self.assertGreater(a.now(),0)

    def test_wall_adapter_requires_no_ros_robot_node(self):
        from sim_time import install_client
        from types import SimpleNamespace
        client=SimpleNamespace(robot=SimpleNamespace())
        install_client(client,False)
        self.assertFalse(client.eval_clock.use_sim_time)
        self.assertEqual(client.control_loop.__self__,client)

    def test_manifest_budget_and_pairing(self):
        m=manifest();self.assertEqual(len(m['per_checkpoint']),30);self.assertEqual(len(m['best_checkpoint_only']),102)
        for color in ('red','green','blue'):
            self.assertEqual(sum(x['color']==color for x in m['per_checkpoint']),10)
        off=[r for r in m['best_checkpoint_only'] if r['suite']=='main-rtc-off']
        self.assertEqual([(r['seed'],r['color'],r['task']) for r in off],[(r['seed'],r['color'],r['task']) for r in m['per_checkpoint']])
        for row in m['per_checkpoint']+m['best_checkpoint_only']:
            args=command(row,Path('/test/checkpoint'),Path('/test/out'),Path('/test/bank'))
            self.assertIn('--sim-time',args);self.assertIn('/joint_states_measured',args)
            self.assertEqual('--no-rtc' in args,not row['rtc'])


class FrozenSceneTests(unittest.TestCase):
    def test_frozen_bank_refuses_missing_and_mutated_inputs(self):
        from phase3 import validate_bank,SEEDS,COLORS
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as name:
            bank=Path(name)
            with self.assertRaises(FileNotFoundError):validate_bank(bank)
            files={}
            for file in ['baseline.json']+[f'seed-{seed}.json' for seed in SEEDS]:
                (bank/file).write_text('{}\n');files[file]=hashlib.sha256((bank/file).read_bytes()).hexdigest()
            (bank/'bank-manifest.json').write_text(json.dumps(dict(seeds=SEEDS,colors=COLORS,files_sha256=files)))
            import phase3
            from unittest.mock import patch
            digest=hashlib.sha256((bank/'bank-manifest.json').read_bytes()).hexdigest()
            with patch.object(phase3,'BANK_SHA256',digest):
                self.assertEqual(len(validate_bank(bank)),64)
            (bank/'seed-900000.json').write_text('{"mutated":true}')
            with self.assertRaises(ValueError):validate_bank(bank)

if __name__=='__main__':unittest.main(verbosity=2)
