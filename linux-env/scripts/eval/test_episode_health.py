"""Native log/RPC faults cannot be accepted as inference starvation."""
import importlib.util,unittest
from types import SimpleNamespace

class Reader:
    def __init__(self,*parts):self.parts=list(parts)
    def read(self):return self.parts.pop(0) if self.parts else ''

@unittest.skipUnless(importlib.util.find_spec('rclpy'),'Use deployment pixi environment')
class HealthTests(unittest.TestCase):
    def test_split_server_error_fails(self):
        from run_episodes import inference_health
        process=SimpleNamespace(poll=lambda:None)
        streams=(Reader('INFO ok\nError in Stream','Actions: simulated model failure\n'),Reader())
        tails=['',''];inference_health(process,streams,tails)
        with self.assertRaises(RuntimeError):inference_health(process,streams,tails)
    def test_receive_error_fails_even_without_server_error(self):
        from run_episodes import inference_health
        with self.assertRaises(RuntimeError):inference_health(SimpleNamespace(poll=lambda:None),(Reader(),Reader('Error receiving actions: simulated lost RPC')),['',''])
    def test_dead_server_fails_before_ready_timeout(self):
        from run_episodes import inference_health
        with self.assertRaises(RuntimeError):inference_health(SimpleNamespace(poll=lambda:70,returncode=70),(Reader(),Reader()),['',''])

if __name__=='__main__':unittest.main(verbosity=2)
