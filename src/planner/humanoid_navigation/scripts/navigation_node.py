#!/usr/bin/env python3
"""Asynchronous four-dimensional planner and 50 Hz command server."""
import math
import threading
import time
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Bool
from visualization_msgs.msg import MarkerArray
from humanoid_navigation.msg import HumanoidCommand, HeightState, RawVoxelMap
from humanoid_navigation.motion import Tracker
from humanoid_navigation.core import State, Search, Trajectory, body_velocity, clamp, wrap
from humanoid_navigation.ros_support import configuration, quaternion, yaw_of, map_markers


class Navigation:
    def __init__(self):
        self.world, self.body, self.limits, _, self.goal = configuration()
        self.lock = threading.RLock()
        self.odom = self.height = None
        self.odom_time = self.height_time = -math.inf
        self.state, self.trajectory = 'WAITING', None
        self.generation, self.worker = 0, None
        self.last_command = [0., 0., 0.]
        self.tracker = Tracker(self.limits)
        self.command_h = None
        self.last_tick, self.last_check = time.monotonic(), 0.
        self.timeout = rospy.get_param('~feedback_timeout', .5)
        self.command_pub = rospy.Publisher('command', HumanoidCommand, queue_size=1)
        self.status_pub = rospy.Publisher('status', String, queue_size=1, latch=True)
        self.execution_collision = False
        self.path_pub = rospy.Publisher('path', Path, queue_size=1, latch=True)
        self.map_pub = rospy.Publisher('obstacles', MarkerArray, queue_size=1, latch=True)
        self.map_pub.publish(map_markers(self.world))
        rospy.Subscriber('odom', Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber('height', HeightState, self.on_height, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber('raw_map', RawVoxelMap, self.on_map, queue_size=1)
        rospy.Subscriber('executor_collision', Bool, self.on_collision, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(.02), self.tick)

    def status(self, value):
        if value != self.state:
            rospy.loginfo('Humanoid: %s', value)
        self.state = value
        self.status_pub.publish(value)

    def on_collision(self, msg):
        if msg.data:
            with self.lock:
                self.execution_collision = True

    def on_map(self, msg):
        from humanoid_navigation.core import VoxelMap
        try:
            if msg.header.frame_id != 'world':
                raise ValueError('Map must be in world frame')
            world = VoxelMap.from_raw(msg.origin,msg.shape,msg.resolution,msg.data)
        except ValueError as error:
            rospy.logerr('Invalid map: %s',error)
            with self.lock:
                self.generation += 1
                self.trajectory = None
                self.status('FAULT: invalid_map')
            return
        with self.lock:
            self.world = world
            self.generation += 1
            self.trajectory = None
            self.map_pub.publish(map_markers(world))
            self.status('WAITING')

    def on_odom(self, msg):
        with self.lock:
            p, r = msg.pose.pose.position, msg.pose.pose.orientation
            values = (p.x, p.y, r.x, r.y, r.z, r.w)
            if msg.header.frame_id != 'world' or not all(math.isfinite(v) for v in values):
                return
            if abs(sum(v*v for v in values[2:])-1) > .05:
                return
            if abs((rospy.Time.now()-msg.header.stamp).to_sec()) > self.timeout:
                return
            self.odom, self.odom_time = (p.x, p.y, yaw_of(r)), time.monotonic()

    def on_height(self, msg):
        with self.lock:
            if (not math.isfinite(msg.h) or not self.body.hmin <= msg.h <= self.body.hmax
                    or abs((rospy.Time.now()-msg.header.stamp).to_sec()) > self.timeout):
                return
            self.height, self.height_time = msg.h, time.monotonic()

    def on_goal(self, msg):
        with self.lock:
            p, r = msg.pose.position, msg.pose.orientation
            if msg.header.frame_id != 'world' or not all(math.isfinite(v) for v in (p.x,p.y,r.x,r.y,r.z,r.w)):
                rospy.logwarn('Goal must be finite and in world frame')
                return
            if abs(r.x*r.x+r.y*r.y+r.z*r.z+r.w*r.w-1) > .05:
                return
            self.goal = State(p.x, p.y, yaw_of(r), self.body.hmax)
            self.generation += 1
            self.trajectory = None
            self.status('WAITING')

    def current(self):
        return State(*self.odom, self.height)

    def launch_plan(self, q):
        if self.worker is not None and self.worker.is_alive():
            return
        generation, goal, world = self.generation, self.goal, self.world
        self.status('PLANNING')

        def work():
            try:
                search = Search(world, self.body, self.limits,
                                xy=rospy.get_param('~xy_resolution', .05),
                                yaw=math.radians(rospy.get_param('~yaw_resolution_deg', 15.)),
                                dh=rospy.get_param('~height_resolution', .05),
                                timeout=rospy.get_param('~search_timeout', 30.),
                                max_expansions=rospy.get_param('~max_expansions', 200000))
                path = search.plan(q, goal, lambda: generation != self.generation or rospy.is_shutdown())
                traj = Trajectory(path, world, self.body, self.limits) if path else None
                with self.lock:
                    if generation != self.generation:
                        return
                    if traj is None:
                        self.status('FAULT: '+search.reason)
                        return
                    # The robot was held still while planning. Refuse a stale start.
                    actual = self.current()
                    if (math.hypot(actual.x-q.x, actual.y-q.y) > .03 or abs(wrap(actual.yaw-q.yaw)) > .04
                            or abs(actual.h-q.h) > .02):
                        self.status('WAITING')
                        return
                    self.trajectory, self.started = traj, time.monotonic()
                    message = Path()
                    message.header.frame_id, message.header.stamp = 'world', rospy.Time.now()
                    for point in path:
                        pose = PoseStamped()
                        pose.header = message.header
                        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point.x, point.y, point.h
                        pose.pose.orientation = quaternion(point.yaw)
                        message.poses.append(pose)
                    self.path_pub.publish(message)
                    self.status('EXECUTING')
            except Exception as error:
                rospy.logerr('Planning error: %s', error)
                with self.lock:
                    if generation == self.generation:
                        self.status('FAULT: planning_exception')

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def stop(self, now):
        if self.height is not None:
            self.last_command, self.command_h = self.tracker.stop(self.height)
        if self.command_h is not None:
            self.publish(now, self.last_command, self.command_h)

    def publish(self, now, velocity, h):
        msg = HumanoidCommand()
        msg.header.stamp, msg.header.frame_id = now, 'base_link'
        msg.vx, msg.vy, msg.wz = velocity
        msg.h = h
        self.command_pub.publish(msg)

    def tick(self, event):
        with self.lock:
            clock, now = time.monotonic(), rospy.Time.now()
            dt = clamp(clock-self.last_tick, .001, .05)
            self.last_tick = clock
            if self.odom is None or self.height is None:
                return
            if clock-min(self.odom_time, self.height_time) > self.timeout:
                self.generation += 1
                self.trajectory = None
                self.status('FAULT: feedback_timeout')
                self.stop(now)
                return
            q = self.current()
            if self.world.collision(q, self.body) or self.execution_collision:
                self.generation += 1
                self.trajectory = None
                self.status('FAULT: current_collision')
            if self.state == 'WAITING':
                self.launch_plan(q)
            if self.state != 'EXECUTING':
                self.stop(now)
                return
            t = clock-self.started
            if clock-self.last_check > .1:
                self.last_check = clock
                if not self.trajectory.remaining_safe(t, self.world, self.body):
                    self.generation += 1
                    self.trajectory = None
                    self.status('WAITING')
                    self.stop(now)
                    return
            target, v = self.trajectory.sample(t)
            if math.hypot(target.x-q.x, target.y-q.y) > .2 or abs(wrap(target.yaw-q.yaw)) > .35 or abs(target.h-q.h) > .08:
                self.generation += 1
                self.trajectory = None
                self.status('WAITING')
                self.stop(now)
                return
            if t >= self.trajectory.total and math.hypot(q.x-self.goal.x,q.y-self.goal.y) < .1 and abs(wrap(q.yaw-self.goal.yaw)) < math.radians(5) and abs(q.h-self.goal.h) < .02:
                self.status('ARRIVED')
                self.stop(now)
                return
            self.last_command, self.command_h = self.tracker.command(q,target,v,dt)
            self.publish(now, self.last_command, self.command_h)


if __name__ == '__main__':
    rospy.init_node('humanoid_navigation')
    Navigation()
    rospy.spin()
