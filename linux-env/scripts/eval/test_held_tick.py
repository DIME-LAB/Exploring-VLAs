import copy,math,sys,unittest
from contextlib import nullcontext
from held_tick import HeldTick,float32,verify_ack,verify_step
class Generation:
 def own(self,_):return nullcontext()
 def require(self,epoch,stamp):
  if epoch!=2 or stamp<=0:raise RuntimeError('Retired observation')
class Sensors:
 def __init__(self):self.current=dict(epoch=2,sim_ns=1_000_000_000);self.reads=0;self.sent=[];self.ack_transform=lambda x:x;self.bad_step=0
 def read(self,timeout=30):self.reads+=1;return self.current['sim_ns'],dict(tick=(copy.deepcopy(self.current),None))
 def publish(self,payload):
  self.sent.append(payload);self.ack=dict(epoch=payload['epoch'],observation_stamp_ns=payload['observation_stamp_ns'],application_stamp_ns=payload['observation_stamp_ns'],applied_command=float32(payload['command']))
  self.current['sim_ns']+=33_333_333+self.bad_step
 def read_ack(self,epoch,stamp):return self.ack_transform(self.ack)
class Tests(unittest.TestCase):
 def setUp(self):self.sensors=Sensors();self.session=HeldTick(Generation(),self.sensors,self.sensors.publish);self.session.read()
 def test_each_command_applies_at_current_tick_then_advances_once(self):
  for i in range(3):
   proof,next_snapshot=self.session.step([i/10]*6)
   self.assertTrue(proof['command']['advance']);self.assertEqual(proof['ack']['applied_command'],float32([i/10]*6));self.assertEqual(next_snapshot['sim_ns'],1_000_000_000+(i+1)*33_333_333)
  self.assertEqual(len(self.sensors.sent),3);self.assertEqual(self.sensors.reads,4)
 def test_bad_action_cannot_publish_or_advance(self):
  for values in ([0]*5,[0]*5+[math.nan],[0]*5+[1e99]):
   with self.assertRaises(ValueError):self.session.step(values)
  self.assertEqual(self.sensors.sent,[])
 def test_wrong_ack_fails_before_reading_next_frame(self):
  for change in (dict(epoch=9),dict(application_stamp_ns=0),dict(applied_command=[.1]*6),dict(error='bad')):
   self.setUp()
   self.sensors.ack_transform=lambda ack,change=change:dict(ack,**change)
   with self.assertRaises(RuntimeError):self.session.step([0]*6)
   self.assertIsNone(self.session.current)
   with self.assertRaisesRegex(RuntimeError,'new reset epoch'):self.session.read()
  self.assertEqual(self.sensors.reads,2)
 def test_double_advance_is_refused(self):
  self.sensors.bad_step=33_333_333
  with self.assertRaisesRegex(RuntimeError,'exactly one'):self.session.step([0]*6)
 def test_frame_epoch_is_refused(self):
  with self.assertRaisesRegex(RuntimeError,'reset epoch'):verify_step(dict(sim_ns=1),dict(epoch=9,sim_ns=33_333_334),2)
if __name__=='__main__':unittest.main()
