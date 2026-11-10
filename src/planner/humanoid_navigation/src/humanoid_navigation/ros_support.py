"""Shared ROS parameter loading and visualization."""
import math
import rospy
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from .core import Body, Limits, scene


def configuration():
    body = Body(rospy.get_param('~body_table', None), rospy.get_param('~margin', .03))
    limits = Limits(**rospy.get_param('~limits', {}))
    world, start, goal = scene(rospy.get_param('~scene', 'compound'))
    from .core import VoxelMap, State
    from humanoid_navigation.msg import RawVoxelMap
    snapshot = rospy.wait_for_message('raw_map', RawVoxelMap, timeout=30.)
    if snapshot.header.frame_id != 'world':
        raise ValueError('Map must be in world frame')
    world = VoxelMap.from_raw(snapshot.origin,snapshot.shape,snapshot.resolution,snapshot.data)
    start = State(start.x,start.y,start.yaw,body.hmax)
    goal = State(goal.x,goal.y,goal.yaw,body.hmax)
    return world, body, limits, start, goal


def quaternion(yaw):
    return Quaternion(0, 0, math.sin(yaw/2), math.cos(yaw/2))


def yaw_of(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def marker(mid, namespace, color):
    m = Marker()
    m.header.frame_id = 'world'
    m.header.stamp = rospy.Time.now()
    m.ns, m.id, m.type, m.action = namespace, mid, Marker.CUBE, Marker.ADD
    m.pose.orientation.w = 1
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def body_markers(q, body):
    result = MarkerArray()
    for i, (x, y, z, lx, ly, lz) in enumerate(body.boxes(q)):
        m = marker(i, 'body', (.2, .7, 1., .7))
        c, s = math.cos(q.yaw), math.sin(q.yaw)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = q.x+c*x-s*y, q.y+s*x+c*y, z
        m.pose.orientation = quaternion(q.yaw)
        m.scale.x, m.scale.y, m.scale.z = lx+2*body.margin, ly+2*body.margin, lz+2*body.margin
        result.markers.append(m)
    m = marker(100, 'com', (1., .2, .2, 1.))
    m.type = Marker.SPHERE
    m.pose.position.x, m.pose.position.y, m.pose.position.z = q.x, q.y, q.h
    m.scale.x = m.scale.y = m.scale.z = .07
    result.markers.append(m)
    m = marker(101, 'height', (1., 1., 1., 1.))
    m.type = Marker.TEXT_VIEW_FACING
    m.pose.position.x, m.pose.position.y, m.pose.position.z = q.x, q.y, 1.7
    m.scale.z = .12
    m.text = 'CoM h = %.3f m' % q.h
    result.markers.append(m)
    return result


def map_markers(world):
    result = MarkerArray()
    clear = marker(0,'obstacles',(0.,0.,0.,0.))
    clear.action = Marker.DELETEALL
    result.markers.append(clear)
    for i, b in enumerate(world.obstacles):
        m = marker(i, 'obstacles', (.6, .6, .6, 1.))
        m.pose.position.x, m.pose.position.y, m.pose.position.z = [(b[j]+b[j+3])/2 for j in range(3)]
        m.scale.x, m.scale.y, m.scale.z = [b[j+3]-b[j] for j in range(3)]
        result.markers.append(m)
    m = marker(1000, 'floor', (.2, .25, .2, .4))
    m.pose.position.z = -.025
    m.scale.x, m.scale.y, m.scale.z = 6, 4, .05
    result.markers.append(m)
    return result
