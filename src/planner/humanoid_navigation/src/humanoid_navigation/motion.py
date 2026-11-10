"""ROS independent command tracking and kinematic execution used by the demo."""
import math
from .core import State, body_velocity, clamp, wrap


class Tracker:
    def __init__(self, limits):
        self.limits = limits
        self.velocity = [0.,0.,0.]
        self.h = None

    def stop(self, height):
        self.velocity = [0.,0.,0.]
        self.h = height
        return self.velocity, self.h

    def command(self, q, target, derivative, dt):
        l = self.limits
        vx,vy = body_velocity(derivative[0]+clamp(target.x-q.x,-.1,.1),
                              derivative[1]+clamp(target.y-q.y,-.1,.1),q.yaw)
        desired = (vx,vy,derivative[2]+clamp(wrap(target.yaw-q.yaw),-.15,.15))
        self.velocity = [clamp(clamp(value,-speed,speed),old-acc*dt,old+acc*dt)
                         for value,old,speed,acc in zip(desired,self.velocity,(l.vx,l.vy,l.wz),(l.ax,l.ay,l.aw))]
        if self.h is None:
            self.h = q.h
        self.h = clamp(target.h,self.h-l.vh*dt,self.h+l.vh*dt)
        return self.velocity, self.h


class Kinematics:
    def __init__(self, limits, body):
        self.limits, self.body, self.vh = limits, body, 0.

    def step(self, q, velocity, height, dt, valid=True):
        l = self.limits
        if not valid:
            self.vh = 0.
            return q
        vx,vy,wz = (clamp(v,-m,m) for v,m in zip(velocity,(l.vx,l.vy,l.wz)))
        dh = clamp(height,self.body.hmin,self.body.hmax)-q.h
        # A hold command cancels the vertical filter immediately, including
        # emergency stops. Normal motion obeys the configured acceleration.
        if abs(dh) < 1e-8:
            self.vh = 0.
        else:
            self.vh = clamp(clamp(4*dh,-l.vh,l.vh),self.vh-l.ah*dt,self.vh+l.ah*dt)
        h = clamp(q.h+self.vh*dt,self.body.hmin,self.body.hmax)
        c,s = math.cos(q.yaw),math.sin(q.yaw)
        return State(q.x+(c*vx-s*vy)*dt,q.y+(s*vx+c*vy)*dt,q.yaw+wz*dt,h)
