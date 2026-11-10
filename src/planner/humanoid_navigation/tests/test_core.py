"""Run directly on Windows/Linux, or through catkin/CTest (no ROS required)."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from humanoid_navigation.core import Body, Limits, Search, State, Trajectory, VoxelMap, body_velocity, scene, wrap
from humanoid_navigation.motion import Tracker, Kinematics


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.body = Body()

    def test_height_changes_shape(self):
        low = self.body.boxes(State(0,0,0,.55))
        high = self.body.boxes(State(0,0,0,.85))
        mid = self.body.boxes(State(0,0,0,.70))
        self.assertAlmostEqual(max(b[2]+b[5]/2 for b in low), 1.1)
        self.assertAlmostEqual(max(b[2]+b[5]/2 for b in high), 1.45)
        self.assertNotEqual(low[0][5], high[0][5])
        self.assertAlmostEqual(mid[0][5], (low[0][5]+high[0][5])/2)

    def test_unknown_and_bounds(self):
        w = VoxelMap(unknown=[(-.1,-.1,.5,.1,.1,1.)])
        self.assertTrue(w.collision(State(0,0,0,.85),self.body))
        self.assertTrue(w.collision(State(3,0,0,.85),self.body))
        self.assertFalse(VoxelMap().collision(State(0,0,0,.85),self.body))
        self.assertTrue(w.collision(State(0,0,0,.9),self.body))

    def test_rotation_swept_collision(self):
        w = VoxelMap(obstacles=[(.20,.25,.9,.25,.30,1.1)])
        a,b = State(0,0,0,.85), State(0,0,math.pi,.85)
        self.assertFalse(w.collision(a,self.body))
        self.assertFalse(w.collision(b,self.body))
        self.assertFalse(w.edge_free(a,b,self.body))

    def test_translation_swept_collision(self):
        w = VoxelMap(obstacles=[(-.025,-.1,.8,.025,.1,1.)])
        self.assertFalse(w.edge_free(State(-1,0,0,.85),State(1,0,0,.85),self.body))

    def test_frame_and_wrap(self):
        vx,vy = body_velocity(1,0,math.pi/2)
        self.assertAlmostEqual(vx,0)
        self.assertAlmostEqual(vy,-1)
        self.assertAlmostEqual(wrap(math.radians(-358)),math.radians(2))

    def test_raw_snapshot_preserves_unknown_and_occupied(self):
        data = [0]*64
        data[(1*4+1)*4+1] = 1
        data[(2*4+2)*4+2] = -1
        w = VoxelMap.from_raw((0,0,0),(4,4,4),.5,data)
        self.assertEqual(w.obstacles,[(.5,.5,.5,1.,1.,1.)])
        self.assertEqual(w.unknown,[(1.,1.,1.,1.5,1.5,1.5)])
        full = VoxelMap.from_raw((0,0,0),(4,4,4),.5,[1]*64)
        self.assertEqual(full.obstacles,[(0.,0.,0.,2.,2.,2.)])

    def test_low_beam_forbids_standing(self):
        w,_,_ = scene('low')
        self.assertTrue(w.collision(State(0,0,0,.85),self.body))
        self.assertFalse(w.collision(State(0,0,0,.55),self.body))
        self.assertFalse(w.edge_free(State(0,0,0,.55), State(0,0,0,.85), self.body))


class PlanningTests(unittest.TestCase):
    def test_scenes_and_timing(self):
        for name in ('open','narrow','low','compound'):
            with self.subTest(scene=name):
                w,a,b = scene(name)
                body,limits = Body(),Limits()
                p = Search(w,body,limits).plan(a,b)
                self.assertIsNotNone(p)
                t = Trajectory(p,w,body,limits)
                previous, previous_hv = None, None
                dt = .01
                for i in range(math.ceil(t.total/dt)+1):
                    q,v = t.sample(i*dt)
                    self.assertFalse(w.collision(q,body))
                    bv = (*body_velocity(v[0],v[1],q.yaw),v[2],v[3])
                    for value,limit in zip(bv,(limits.vx,limits.vy,limits.wz,limits.vh)):
                        self.assertLessEqual(abs(value),limit+1e-8)
                    if previous is not None:
                        for value,old,limit in zip(bv,previous,(limits.ax,limits.ay,limits.aw,limits.ah)):
                            self.assertLessEqual(abs(value-old)/dt,limit+1e-5)
                    previous = bv
                self.assertEqual(t.sample(t.total)[0],b)
                self.assertTrue(t.remaining_safe(0,w,body))
                if name in ('narrow','compound'):
                    self.assertTrue(any(abs(math.sin(q.yaw))>.9 for q in p))
                if name in ('low','compound'):
                    self.assertLess(min(q.h for q in p),.7)

    def test_unwrapped_trajectory(self):
        a,b = State(0,0,math.radians(179),.85),State(0,0,math.radians(-179),.85)
        t = Trajectory([a,b],VoxelMap(),Body())
        self.assertLess(t.total,1)
        self.assertAlmostEqual(abs(t.sample(t.total/2)[0].yaw),math.pi)

    def test_budget_and_cancel(self):
        w,a,b = scene('blocked')
        search = Search(w,Body(),max_expansions=50,timeout=.5)
        self.assertIsNone(search.plan(a,b))
        self.assertIn(search.reason,('no_path','cancelled_or_budget_exceeded'))
        self.assertIsNone(search.plan(a,b,lambda: True))

    def test_astar_fallback_around_obstacle(self):
        # A diagonal wall defeats a single constant transit-posture connection.
        w = VoxelMap(bounds=(-1,-1,0,1,1,2),obstacles=[(-.1,-.25,0,.1,.25,2)])
        a,b = State(-.65,0,0,.85),State(.65,0,0,.85)
        body = Body()
        s = Search(w,body,xy=.15,yaw=math.pi/2,dh=.3,timeout=10)
        p = s.plan(a,b)
        self.assertIsNotNone(p,s.reason)
        self.assertTrue(any(abs(q.y)>.3 for q in p))
        for u,v in zip(p,p[1:]):
            self.assertTrue(w.edge_free(u,v,body))

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            Limits(vx=0)

    def test_closed_loop_scenes(self):
        for name in ('open','narrow','low','compound'):
            with self.subTest(scene=name):
                w,q,goal = scene(name)
                body,l = Body(),Limits()
                trajectory = Trajectory(Search(w,body,l).plan(q,goal),w,body,l)
                tracker,executor = Tracker(l),Kinematics(l,body)
                dt = .02
                last = (0.,0.,0.)
                for i in range(math.ceil((trajectory.total+5)/dt)):
                    target,derivative = trajectory.sample(i*dt)
                    velocity,h = tracker.command(q,target,derivative,dt)
                    for v,old,limit in zip(velocity,last,(l.ax,l.ay,l.aw)):
                        self.assertLessEqual(abs(v-old)/dt,limit+1e-8)
                    nq = executor.step(q,velocity,h,dt)
                    self.assertTrue(w.edge_free(q,nq,body),str((name,i,q,nq)))
                    q,last = nq,velocity
                position_error = math.hypot(q.x-goal.x,q.y-goal.y)
                yaw_error = abs(wrap(q.yaw-goal.yaw))
                self.assertLess(position_error,.1)
                self.assertLess(yaw_error,math.radians(5))
                self.assertLess(abs(q.h-goal.h),.02)
                print('%s closed loop: position=%.6fm yaw=%.6fdeg height=%.6fm' %
                      (name,position_error,math.degrees(yaw_error),abs(q.h-goal.h)))

    def test_fault_stop_holds_crouch_and_watchdog(self):
        body,l = Body(),Limits()
        tracker,executor = Tracker(l),Kinematics(l,body)
        q = State(0,0,.5,.6)
        executor.vh = .1
        velocity,h = tracker.stop(q.h)
        self.assertEqual(velocity,[0.,0.,0.])
        self.assertEqual(executor.step(q,velocity,h,.02),q)
        self.assertEqual(executor.step(q,(.4,.2,.5),.85,.02,valid=False),q)

    def test_updated_map_invalidates_trajectory(self):
        w,a,b = scene('open')
        body = Body()
        t = Trajectory([a,b],w,body)
        blocked,_,_ = scene('blocked')
        self.assertFalse(t.remaining_safe(0,blocked,body))


if __name__ == '__main__':
    unittest.main(verbosity=2)
