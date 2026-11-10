#!/usr/bin/env python3
"""Kinematic test executor, deliberately independent of a locomotion policy."""
import math
import threading
import time
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from humanoid_navigation.msg import HumanoidCommand, HeightState
from humanoid_navigation.core import State, clamp
from humanoid_navigation.motion import Kinematics
from humanoid_navigation.ros_support import configuration, quaternion, body_markers


class Executor:
    def __init__(self):
        self.world, self.body, self.limits, self.q, _ = configuration()
        self.lock = threading.RLock()
        self.command = None
        self.received = -math.inf
        self.last = time.monotonic()
        self.vh = 0.
        self.kinematics = Kinematics(self.limits,self.body)
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=1)
        self.height_pub = rospy.Publisher('height', HeightState, queue_size=1)
        self.body_pub = rospy.Publisher('body', MarkerArray, queue_size=1)
        self.collision_pub = rospy.Publisher('executor_collision', Bool, queue_size=1, latch=True)
        self.collision_pub.publish(False)
        rospy.Subscriber('command', HumanoidCommand, self.on_command, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(.02), self.tick)

    def on_command(self, msg):
        with self.lock:
            if (msg.header.frame_id != 'base_link' or not all(math.isfinite(v) for v in (msg.vx,msg.vy,msg.wz,msg.h))
                    or abs((rospy.Time.now()-msg.header.stamp).to_sec()) > .25):
                return
            self.command, self.received = msg, time.monotonic()

    def tick(self, event):
        with self.lock:
            clock, stamp = time.monotonic(), rospy.Time.now()
            dt, self.last = clamp(clock-self.last, .001, .05), clock
            q, l = self.q, self.limits
            vx = vy = wz = 0.
            h = q.h
            if self.command is not None and clock-self.received < .25:
                cmd = self.command
                vx, vy, wz = clamp(cmd.vx,-l.vx,l.vx), clamp(cmd.vy,-l.vy,l.vy), clamp(cmd.wz,-l.wz,l.wz)
                h = cmd.h
            nq = self.kinematics.step(q,(vx,vy,wz),h,dt, self.command is not None and clock-self.received < .25)
            # Do not hide execution collisions by silently clipping motion.
            if not self.world.edge_free(q, nq, self.body):
                self.collision_pub.publish(True)
                rospy.logerr_throttle(1., 'EXECUTOR COLLISION: test failed; holding pose')
                self.kinematics.vh = 0.
                vx = vy = wz = 0.
            else:
                self.q = nq
            q = self.q
            odom = Odometry()
            odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'world', 'base_link'
            odom.pose.pose.position.x, odom.pose.pose.position.y = q.x, q.y
            odom.pose.pose.orientation = quaternion(q.yaw)
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = vx, vy, wz
            self.odom_pub.publish(odom)
            height = HeightState()
            height.header.stamp, height.header.frame_id, height.h = stamp, 'world', q.h
            self.height_pub.publish(height)
            self.body_pub.publish(body_markers(q, self.body))


if __name__ == '__main__':
    rospy.init_node('humanoid_executor')
    Executor()
    rospy.spin()
