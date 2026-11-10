#!/usr/bin/env python3
"""End-to-end ROS transport, execution and fault acceptance tests."""
import math
import time
import unittest
import rospy
import rostest
import rosnode
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from humanoid_navigation.msg import HumanoidCommand, HeightState, RawVoxelMap
from humanoid_navigation.core import scene


class Acceptance(unittest.TestCase):
    def test_navigation(self):
        self.status = ''
        self.odom = self.height = self.command = None
        self.collided = False
        rospy.Subscriber('/humanoid/status',String,lambda m: setattr(self,'status',m.data))
        rospy.Subscriber('/humanoid/odom',Odometry,lambda m: setattr(self,'odom',m))
        rospy.Subscriber('/humanoid/height',HeightState,lambda m: setattr(self,'height',m.h))
        rospy.Subscriber('/humanoid/command',HumanoidCommand,lambda m: setattr(self,'command',m))
        rospy.Subscriber('/humanoid/executor_collision',Bool,lambda m: setattr(self,'collided',self.collided or m.data))
        mode = rospy.get_param('~mode','arrival')
        deadline = time.monotonic()+65
        killed = False
        held = None
        updated = False
        map_pub = rospy.Publisher('/humanoid/raw_map',RawVoxelMap,queue_size=1,latch=True)
        while time.monotonic()<deadline and not rospy.is_shutdown():
            if mode == 'replan_failure' and not updated and self.status == 'EXECUTING':
                w,_,_ = scene('blocked')
                msg = RawVoxelMap()
                msg.header.frame_id, msg.header.stamp = 'world', rospy.Time.now()
                msg.origin, msg.shape, msg.resolution = [-3.,-2.,0.], [120,80,40], .05
                msg.data = [int(any(b[0] < -3+(x+.5)*.05 < b[3] and b[1] < -2+(y+.5)*.05 < b[4]
                                    and b[2] < (z+.5)*.05 < b[5] for b in w.obstacles))
                            for x in range(120) for y in range(80) for z in range(40)]
                held = self.height
                map_pub.publish(msg)
                updated = True
            if mode == 'feedback_loss' and not killed and self.height is not None and self.height<.7:
                held = self.height
                rosnode.kill_nodes(['/humanoid/executor'])
                killed = True
            if mode == 'arrival' and self.status == 'ARRIVED':
                break
            if mode != 'arrival' and self.status.startswith('FAULT'):
                time.sleep(.15)
                break
            self.assertFalse(self.collided,'Executor reported a swept collision')
            if mode == 'arrival':
                self.assertFalse(self.status.startswith('FAULT'),self.status)
            time.sleep(.02)
        self.assertFalse(self.collided)
        self.assertIsNotNone(self.command)
        if mode == 'arrival':
            self.assertEqual(self.status,'ARRIVED')
            p = self.odom.pose.pose.position
            r = self.odom.pose.pose.orientation
            yaw = math.atan2(2*r.w*r.z,1-2*r.z*r.z)
            self.assertLess(math.hypot(p.x-1.2,p.y),.1)
            self.assertLess(abs(yaw),math.radians(5))
            self.assertLess(abs(self.height-.85),.02)
        else:
            self.assertTrue(self.status.startswith('FAULT'),self.status)
            if mode == 'feedback_loss':
                self.assertTrue(killed)
                self.assertIn('feedback_timeout',self.status)
                self.assertLess(abs(self.command.h-held),.01)
            elif mode == 'replan_failure':
                self.assertTrue(updated)
                self.assertLess(abs(self.command.h-held),.01)
            else:
                self.assertAlmostEqual(self.command.h,.85,places=3)
        self.assertAlmostEqual(self.command.vx,0.)
        self.assertAlmostEqual(self.command.vy,0.)
        self.assertAlmostEqual(self.command.wz,0.)


if __name__ == '__main__':
    rospy.init_node('humanoid_acceptance')
    rostest.rosrun('humanoid_navigation','acceptance',Acceptance)
