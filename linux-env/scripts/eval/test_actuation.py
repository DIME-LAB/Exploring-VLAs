from types import SimpleNamespace as N
import unittest
from actuation import install
class Tests(unittest.TestCase):
 def robot(self):
  seen=[];pub=N(publish=lambda msg:seen.append(msg));robot=N(config=N(actuate=True,use_degrees=True,action_duration_s=1/30),_arm_command_pub=pub,_gripper_command_pub=N(publish=lambda msg:None));return robot,seen
 def msg(self,duration):return N(header=N(stamp=N(sec=0,nanosec=0)),points=[N(positions=[.1,.2],time_from_start=N(sec=0,nanosec=round(duration*1e9)))])
 def test_real_hardware_clock_cannot_enable_immediate_mode(self):
  robot,_=self.robot();original=robot._arm_command_pub.publish
  with self.assertRaises(ValueError):install(robot,'immediate',False)
  self.assertEqual(robot.config.action_duration_s,1/30);self.assertEqual(robot._arm_command_pub.publish,original)
 def test_immediate_native_payload_and_restore(self):
  robot,seen=self.robot();original=robot._arm_command_pub.publish;restore=install(robot,'immediate',True);self.assertEqual(robot.config.action_duration_s,0);msg=self.msg(robot.config.action_duration_s);robot._arm_command_pub.publish(msg);self.assertIs(seen[-1],msg);self.assertEqual(seen[-1].points[0].positions,[.1,.2]);restore();self.assertEqual(robot.config.action_duration_s,1/30);self.assertEqual(robot._arm_command_pub.publish,original)
 def test_native_mode_keeps_positive_duration(self):
  robot,seen=self.robot();restore=install(robot,'native',True);robot._arm_command_pub.publish(self.msg(robot.config.action_duration_s));self.assertEqual(seen[-1].points[0].time_from_start.nanosec,33333333);restore()
 def test_failed_install_rolls_back_duration_and_publisher(self):
  robot,_=self.robot();original=robot._arm_command_pub.publish;robot._gripper_command_pub=None
  with self.assertRaises(AttributeError):install(robot,'immediate',True)
  self.assertEqual(robot.config.action_duration_s,1/30);self.assertEqual(robot._arm_command_pub.publish,original)
if __name__=='__main__':unittest.main()
