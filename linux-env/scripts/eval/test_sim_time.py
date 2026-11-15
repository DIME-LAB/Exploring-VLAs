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
        self.assertEqual(client.eval_observe_every_actions,10)

    def test_diagnostic_cadence_runs_through_the_real_paced_loop(self):
        import sys, threading, time
        from queue import Queue
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        from sim_time import paced_control_loop
        native=ModuleType('lerobot.async_inference.robot_client')
        native._configure_dispatch_thread=lambda config: {'test':True}
        for cadence in (10,25,50):
            class Clock:
                use_sim_time=True
                value=0.
                def now(self):return self.value
                def wait_until(self,deadline,stop):
                    time.sleep(.002)
                    self.value=deadline
                    return None if stop.is_set() else deadline
            client=SimpleNamespace(eval_clock=Clock(),eval_observe_every_actions=cadence,
                start_barrier=threading.Barrier(1),_observation_process=None,running=True,
                config=SimpleNamespace(dispatch_fifo_priority=0,fps=30),latest_action=-1,
                action_queue=Queue(),must_go=threading.Event())
            def action(verbose):
                client.latest_action+=1;client._last_dispatch_cost_s=.0004
                client._last_queue_wait_s=0.;client._last_dispatch_time=time.perf_counter()
            client.control_loop_action=action
            client.control_loop_observation=lambda task,verbose,finish: None
            with patch.dict(sys.modules,{'lerobot.async_inference.robot_client':native}):
                paced_control_loop(client,'test',max_actions=101)
            stats=client.control_loop_stats
            self.assertEqual(stats['actions_sent'],101)
            self.assertEqual(stats['observe_every_actions'],cadence)
            self.assertEqual(stats['missed_action_ticks_after_startup'],0)
            events=stats['observation_cadence_events']
            self.assertEqual(events[0]['executed_actions'],0)
            self.assertGreaterEqual(len(events),101//cadence)
            self.assertTrue(all(e['actions_since_request']>=cadence for e in events[1:]))

    def test_complete_chunks_do_not_retry_or_blend_during_normal_latency(self):
        import sys, threading, time
        from queue import Queue
        from types import SimpleNamespace, ModuleType
        from unittest.mock import patch
        from sim_time import paced_control_loop
        class Clock:
            use_sim_time=True
            value=0.
            def now(self):return self.value
            def wait_until(self,deadline,stop):
                time.sleep(.002);self.value=deadline
                return None if stop.is_set() else deadline
        native=ModuleType('lerobot.async_inference.robot_client');native._configure_dispatch_thread=lambda cfg: {}
        client=SimpleNamespace(eval_clock=Clock(),eval_observe_every_actions=50,
            start_barrier=threading.Barrier(1),_observation_process=None,running=True,
            config=SimpleNamespace(dispatch_fifo_priority=0,fps=30),latest_action=-1,
            action_queue=Queue(),must_go=threading.Event())
        pending=[];requests=[];executed=[]
        def observe(task,verbose,finish):
            requests.append(client.latest_action+1)
            pending.append((client.eval_clock.now()+13/30,client.latest_action+1))
        def action(verbose):
            if pending and client.eval_clock.now()+1e-8>=pending[0][0]:
                _,start=pending.pop(0)
                for step in range(start,start+50):client.action_queue.put(step)
            step=client.action_queue.get_nowait();executed.append(step);client.latest_action=step
            client._last_dispatch_cost_s=.0004;client._last_queue_wait_s=0.;client._last_dispatch_time=time.perf_counter()
        client.control_loop_action=action;client.control_loop_observation=observe
        with patch.dict(sys.modules,{'lerobot.async_inference.robot_client':native}):
            paced_control_loop(client,'test',max_actions=101)
        self.assertEqual(requests,[0,50,100]);self.assertEqual(executed,list(range(101)))
        self.assertEqual(client.control_loop_stats['observation_retries'],0)
        self.assertGreaterEqual(client.control_loop_stats['empty_action_ticks_after_startup'],24)
        self.assertIsNone(client.control_loop_stats['empty_retry_slots'])

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
