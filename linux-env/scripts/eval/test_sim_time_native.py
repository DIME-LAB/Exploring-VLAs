"""CPU-only integration with the installed native LeRobot actuator dequeue path."""
import importlib.util
import math
import threading
import time
import unittest
from queue import Queue
from types import MethodType, SimpleNamespace
from sim_time import PacingClock, paced_control_loop


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Run in the deployment pixi environment')
class NativePacingTests(unittest.TestCase):
    def run_clock(self, rtf, queued=100):
        import torch
        from lerobot.async_inference.robot_client import RobotClient
        from lerobot.async_inference.helpers import TimedAction
        clock = PacingClock.__new__(PacingClock)
        clock.use_sim_time=True;clock.condition=threading.Condition();clock.value=100.
        clock.error=None;clock.received=time.monotonic();clock.stale_wall_s=2
        observations=[];sent=[]
        robot=SimpleNamespace(action_features={'q.pos':float})
        observations_started=threading.Event()
        def send(action):
            if not observations_started.wait(3):raise TimeoutError('No initial observation')
            sent.append((clock.now(),action));return action
        robot.send_action=send
        config=SimpleNamespace(fps=30,dispatch_fifo_priority=0,dispatch_fifo_sudo=False)
        client=SimpleNamespace(config=config,eval_clock=clock,start_barrier=threading.Barrier(1),
            _observation_process=None,action_queue=Queue(),action_queue_lock=threading.Lock(),
            latest_action_lock=threading.Lock(),latest_action=-1,action_queue_size=[],robot=robot,running=True,must_go=threading.Event())
        for i in range(queued):
            client.action_queue.put(TimedAction(timestamp=0,timestep=i,action=torch.tensor([float(i)])))
        client._action_tensor_to_action_dict=MethodType(RobotClient._action_tensor_to_action_dict,client)
        client.control_loop_action=MethodType(RobotClient.control_loop_action,client)
        def observed(*args):
            observations.append(client.latest_action);observations_started.set()
        client.control_loop_observation=observed
        result_errors=[]
        def run():
            try:paced_control_loop(client,'test',max_duration_s=1.)
            except Exception as e:result_errors.append(e)
        worker=threading.Thread(target=run);worker.start()
        # Wait for the first real dequeue instead of sleeping through startup.
        deadline=time.monotonic()+3
        while not sent:
            self.assertLess(time.monotonic(),deadline);threading.Event().wait(.001)
        start=time.monotonic()
        for step in range(1,31):
            remaining=start+step/30/rtf-time.monotonic()
            if remaining>0:threading.Event().wait(remaining)
            clock.feed(100+step/30)
        worker.join(3)
        self.assertFalse(worker.is_alive());self.assertEqual(result_errors,[])
        stats=client.control_loop_stats
        self.assertEqual(stats['missed_action_ticks_after_startup'],0)
        expected=min(30,queued)
        self.assertEqual(stats['actions_sent'],expected)
        self.assertEqual(stats['action_steps'],list(range(expected)))
        self.assertLess(stats['sim_dispatch_jitter_p95_ms'],1e-6)
        if queued>=30:
            self.assertEqual(len(observations),4)
            self.assertEqual(observations[-3:],[9,19,29])
        else:
            self.assertGreater(stats['observation_retries'],0)
            self.assertGreaterEqual(len(observations),3)
        self.assertEqual(len(sent),expected)
        return [round(row['clock_s']-100,7) for row in stats['sim_dispatch_trace']]

    def test_queue_exhaustion_retries_observation(self):
        self.run_clock(1.,queued=15)

    def test_native_dequeue_at_actual_rtf_03_and_10(self):
        self.assertEqual(self.run_clock(.3),self.run_clock(1.))


if __name__=='__main__':unittest.main(verbosity=2)
