"""Flat ground configuration-space planning. No ROS or third-party dependencies."""
import heapq
import itertools
import math
import time
from dataclasses import dataclass

TAU = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def body_velocity(vx, vy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx + s * vy, -s * vx + c * vy


@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float
    h: float


def delta(a, b):
    return (b.x-a.x, b.y-a.y, wrap(b.yaw-a.yaw), b.h-a.h)


def interpolate(a, b, u):
    d = delta(a, b)
    return State(*(v + u*w for v, w in zip((a.x, a.y, a.yaw, a.h), d)))


DEFAULT_TABLE = [
    # h, boxes: [center_x,center_y,center_z,length_x,width_y,height_z]
    [0.55, [[0, 0, .24, .30, .44, .48], [.02, 0, .66, .30, .50, .36],
            [.02, 0, .97, .24, .24, .26]]],
    [0.85, [[0, 0, .39, .24, .44, .78], [0, 0, 1.00, .30, .50, .44],
            [0, 0, 1.335, .24, .24, .23]]],
]


class Body:
    def __init__(self, table=None, margin=.03):
        self.table = table if table is not None else DEFAULT_TABLE
        self.margin = margin
        if len(self.table) < 2 or not math.isfinite(margin) or margin < 0:
            raise ValueError('Need at least two ordered height samples and nonnegative margin')
        count = len(self.table[0][1])
        for i, (h, boxes) in enumerate(self.table):
            if not math.isfinite(h) or (i and h <= self.table[i-1][0]):
                raise ValueError('Height samples must increase')
            if len(boxes) != count or not count:
                raise ValueError('Every height must describe the same body parts')
            if any(len(b) != 6 or not all(math.isfinite(x) for x in b) or min(b[3:]) <= 0 for b in boxes):
                raise ValueError('Invalid body box')
        self.hmin, self.hmax = self.table[0][0], self.table[-1][0]
        self.radius = max(math.hypot(abs(b[0])+b[3]/2+margin, abs(b[1])+b[4]/2+margin)
                          for _, boxes in self.table for b in boxes)
        self.height_lipschitz = max(
            max(abs(b[j]-a[j]) + abs(b[j+3]-a[j+3])/2 for j in range(3))
            / (hi-lo) for (lo, aa), (hi, bb) in zip(self.table, self.table[1:])
            for a, b in zip(aa, bb)) * math.sqrt(3)

    def boxes(self, q):
        h = clamp(q.h, self.hmin, self.hmax)
        for (lo, aa), (hi, bb) in zip(self.table, self.table[1:]):
            if h <= hi:
                u = (h-lo)/(hi-lo)
                return [[x+u*(y-x) for x, y in zip(a, b)] for a, b in zip(aa, bb)]
        raise ValueError('Height outside table')


class VoxelMap:
    """Fully known static voxel map, compressed into occupied rectangular blocks.

    Blocks are snapped outwards to the voxel lattice. Bounds delimit known space;
    unknown blocks allow conservative partial-map tests. The z=0 plane is support.
    """
    def __init__(self, bounds=(-3, -2, 0, 3, 2, 2), resolution=.05, obstacles=(), unknown=()):
        if not math.isfinite(resolution) or resolution <= 0 or not all(math.isfinite(v) for v in bounds) or any(bounds[i] >= bounds[i+3] for i in range(3)):
            raise ValueError('Invalid map geometry')
        self.bounds, self.resolution = tuple(bounds), resolution
        self.obstacles = [self.snap(b) for b in obstacles]
        self.unknown = [self.snap(b) for b in unknown]

    def snap(self, b):
        r = self.resolution
        if len(b) != 6 or not all(math.isfinite(v) for v in b) or any(b[i]>=b[i+3] for i in range(3)):
            raise ValueError('Invalid occupancy block')
        return tuple(math.floor(v/r+1e-9)*r for v in b[:3]) + tuple(math.ceil(v/r-1e-9)*r for v in b[3:])

    def collision(self, q, body):
        if not all(math.isfinite(v) for v in (q.x, q.y, q.yaw, q.h)) or not body.hmin-1e-9 <= q.h <= body.hmax+1e-9:
            return True
        c, s = math.cos(q.yaw), math.sin(q.yaw)
        for x, y, z, lx, ly, lz in body.boxes(q):
            cx, cy = q.x+c*x-s*y, q.y+s*x+c*y
            hx, hy, hz = lx/2+body.margin, ly/2+body.margin, lz/2+body.margin
            ex, ey = abs(c)*hx+abs(s)*hy, abs(s)*hx+abs(c)*hy
            # Only the known supporting plane is exempt from the lower z margin.
            lowz = max(0., z-hz)
            if (cx-ex < self.bounds[0] or cy-ey < self.bounds[1] or lowz < self.bounds[2]
                    or cx+ex > self.bounds[3] or cy+ey > self.bounds[4] or z+hz > self.bounds[5]):
                return True
            for b in itertools.chain(self.obstacles, self.unknown):
                if lowz >= b[5] or z+hz <= b[2]:
                    continue
                bx, by = (b[0]+b[3])/2, (b[1]+b[4])/2
                ax, ay = (b[3]-b[0])/2, (b[4]-b[1])/2
                dx, dy = bx-cx, by-cy
                # Separating axis test: body x/y and world x/y axes.
                if (abs(dx) <= ex+ax and abs(dy) <= ey+ay
                        and abs(c*dx+s*dy) <= hx+abs(c)*ax+abs(s)*ay
                        and abs(-s*dx+c*dy) <= hy+abs(s)*ax+abs(c)*ay):
                    return True
        return False

    def edge_free(self, a, b, body):
        d = delta(a, b)
        travel = math.hypot(d[0], d[1]) + body.radius*abs(d[2]) + body.height_lipschitz*abs(d[3])
        n = max(1, math.ceil(travel/(self.resolution*.5)))
        return all(not self.collision(interpolate(a, b, i/n), body) for i in range(n+1))

    @classmethod
    def from_raw(cls, origin, shape, resolution, data):
        """Losslessly coalesce equal z runs, then y rows and x slabs.

        This consumes plan_env raw log-odds classifications, never inflated data.
        """
        nx, ny, nz = shape
        if min(shape) <= 0 or len(data) != nx*ny*nz or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('Invalid raw snapshot dimensions')
        slabs = {}
        blocks = []
        for x in range(nx):
            rows = {}
            rectangles = []
            for y in range(ny):
                current = set()
                z = 0
                while z < nz:
                    value = data[(x*ny+y)*nz+z]
                    if value not in (-1,0,1):
                        raise ValueError('Invalid occupancy classification')
                    end = z+1
                    while end < nz and data[(x*ny+y)*nz+end] == value:
                        end += 1
                    if value:
                        key = (z,end,value)
                        current.add(key)
                        if key not in rows:
                            rows[key] = y
                    z = end
                for key in list(rows):
                    if key not in current:
                        rectangles.append((rows.pop(key),y,*key))
            rectangles.extend((y,ny,*key) for key,y in rows.items())
            current = set(rectangles)
            for key in list(slabs):
                if key not in current:
                    blocks.append((slabs.pop(key),x,*key))
            for key in current:
                if key not in slabs:
                    slabs[key] = x
        blocks.extend((x,nx,*key) for key,x in slabs.items())
        bounds = tuple(origin) + tuple(origin[i]+shape[i]*resolution for i in range(3))
        world = cls(bounds, resolution)
        for x0,x1,y0,y1,z0,z1,value in blocks:
            box = (origin[0]+x0*resolution,origin[1]+y0*resolution,origin[2]+z0*resolution,
                   origin[0]+x1*resolution,origin[1]+y1*resolution,origin[2]+z1*resolution)
            (world.unknown if value == -1 else world.obstacles).append(box)
        return world


@dataclass
class Limits:
    vx: float = .4
    vy: float = .2
    wz: float = .5
    vh: float = .15
    ax: float = .5
    ay: float = .4
    aw: float = .8
    ah: float = .3

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('All limits must be finite and positive')


class Search:
    def __init__(self, world, body, limits=None, xy=.05, yaw=math.pi/12, dh=.05,
                 max_expansions=200000, timeout=20.):
        if min(xy, yaw, dh) <= 0:
            raise ValueError('Search resolutions must be positive')
        self.world, self.body, self.limits = world, body, limits or Limits()
        self.xy, self.nyaw, self.dh = xy, max(4, round(TAU/yaw)), dh
        self.dyaw = TAU/self.nyaw
        self.nh = math.ceil((body.hmax-body.hmin)/dh)+1
        self.max_expansions, self.timeout = max_expansions, timeout
        self.reason = ''

    def key(self, q):
        return (round(q.x/self.xy), round(q.y/self.xy), round(q.yaw/self.dyaw) % self.nyaw,
                clamp(round((q.h-self.body.hmin)/self.dh), 0, self.nh-1))

    def state(self, k):
        return State(k[0]*self.xy, k[1]*self.xy, k[2]*self.dyaw,
                     min(self.body.hmax, self.body.hmin+k[3]*self.dh))

    def duration(self, a, b):
        dx, dy, dw, dh = delta(a, b)
        # Euclidean/smaller planar speed is conservative for every yaw on an edge.
        l = self.limits
        return max(math.hypot(dx, dy)/min(l.vx, l.vy), abs(dw)/l.wz, abs(dh)/l.vh, .001)

    def plan(self, start, goal, cancelled=lambda: False):
        self.reason = ''
        began = time.monotonic()
        if self.world.collision(start, self.body) or self.world.collision(goal, self.body):
            self.reason = 'start_or_goal_collision'
            return None
        # Analytic expansion: a common transit posture at both ends accelerates
        # simple corridors. Returns a feasible path, not a global optimum.
        best, best_cost = None, math.inf
        for hi in reversed(range(self.nh)):
            h = min(self.body.hmax, self.body.hmin+hi*self.dh)
            for yi in range(self.nyaw):
                if cancelled():
                    self.reason = 'cancelled'
                    return None
                if time.monotonic()-began > self.timeout:
                    self.reason = '' if best is not None else 'cancelled_or_budget_exceeded'
                    return best
                yaw = yi*self.dyaw
                a, b = State(start.x,start.y,yaw,h), State(goal.x,goal.y,yaw,h)
                candidate = [start,a,b,goal]
                cost = sum(self.duration(u,v)*(1+.25*(self.body.hmax-(u.h+v.h)/2)/(self.body.hmax-self.body.hmin))
                           + .02*(abs(delta(u,v)[2])/self.dyaw+abs(delta(u,v)[3])/self.dh)
                           for u,v in zip(candidate,candidate[1:]))
                if cost < best_cost and all(self.world.edge_free(u,v,self.body) for u,v in zip(candidate,candidate[1:])):
                    best, best_cost = candidate, cost
        if best is not None:
            return best
        sk, gk = self.key(start), self.key(goal)
        sq, gq = self.state(sk), self.state(gk)
        if not self.world.edge_free(start, sq, self.body) or not self.world.edge_free(gq, goal, self.body):
            self.reason = 'lattice_connection_collision'
            return None
        # An admissible lower bound in seconds; tie ordering is deterministic.
        def heuristic(q):
            d = delta(q, gq)
            return max(math.hypot(*d[:2])/min(self.limits.vx, self.limits.vy),
                       abs(d[2])/self.limits.wz, abs(d[3])/self.limits.vh)
        serial = itertools.count()
        queue = [(heuristic(sq), next(serial), 0., sk)]
        costs, parents, closed = {sk: 0.}, {}, set()
        translations = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]
        motions = [(x, y, w, h) for x, y in translations for w, h in
                   ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)) if (x, y, w, h) != (0, 0, 0, 0)]
        cache = {}
        def valid(k, q):
            if k not in cache:
                cache[k] = not self.world.collision(q, self.body)
            return cache[k]
        while queue:
            if cancelled() or time.monotonic()-began > self.timeout or len(closed) >= self.max_expansions:
                self.reason = 'cancelled_or_budget_exceeded'
                return None
            _, _, cost, k = heapq.heappop(queue)
            if k in closed or cost > costs[k]:
                continue
            if k == gk:
                keys = [k]
                while keys[-1] != sk:
                    keys.append(parents[keys[-1]])
                path = [start] + [self.state(v) for v in reversed(keys)] + [goal]
                # Shortcut only when the entire four-dimensional edge is safe.
                result = [path[0]]
                i = 0
                while i < len(path)-1:
                    j = len(path)-1
                    while j > i+1 and not self.world.edge_free(path[i], path[j], self.body):
                        j -= 1
                    result.append(path[j])
                    i = j
                return result
            closed.add(k)
            q = self.state(k)
            for dx, dy, dw, dh in motions:
                nk = (k[0]+dx, k[1]+dy, (k[2]+dw) % self.nyaw, k[3]+dh)
                if nk in closed or not 0 <= nk[3] < self.nh:
                    continue
                nq = self.state(nk)
                dt = self.duration(q, nq)
                nc = cost + dt*(1+.25*(self.body.hmax-(q.h+nq.h)/2)/(self.body.hmax-self.body.hmin)) + .02*(abs(dw)+abs(dh))
                if nc >= costs.get(nk, math.inf) or not valid(nk, nq):
                    continue
                if not self.world.edge_free(q, nq, self.body):
                    continue
                costs[nk], parents[nk] = nc, k
                heapq.heappush(queue, (nc+heuristic(nq), next(serial), nc, nk))
        self.reason = 'no_path'
        return None


class Trajectory:
    def __init__(self, path, world, body, limits=None):
        self.segments, self.total = [], 0.
        self.limits = limits or Limits()
        l = self.limits
        for a, b in zip(path, path[1:]):
            if not world.edge_free(a, b, body):
                raise ValueError('Unsafe trajectory edge')
            d = delta(a, b)
            distance = math.hypot(*d[:2])
            if max(abs(v) for v in d) < 1e-10:
                continue
            # Max smoothstep derivative=1.875, second derivative=10/sqrt(3).
            # Rotating body frame adds omega cross velocity to planar acceleration.
            duration = max(.1, 1.875*distance/min(l.vx, l.vy), 1.875*abs(d[2])/l.wz,
                           1.875*abs(d[3])/l.vh,
                           math.sqrt(((10/math.sqrt(3))*distance+1.875**2*distance*abs(d[2]))/min(l.ax,l.ay)),
                           math.sqrt((10/math.sqrt(3))*abs(d[2])/l.aw),
                           math.sqrt((10/math.sqrt(3))*abs(d[3])/l.ah))
            self.segments.append((self.total, duration, a, b))
            self.total += duration
        self.goal = path[-1]

    def sample(self, t):
        for begin, duration, a, b in self.segments:
            if t < begin+duration:
                u = clamp((t-begin)/duration, 0, 1)
                s = 10*u**3-15*u**4+6*u**5
                ds = (30*u**2-60*u**3+30*u**4)/duration
                return interpolate(a, b, s), tuple(v*ds for v in delta(a, b))
        return self.goal, (0., 0., 0., 0.)

    def remaining_safe(self, t, world, body):
        current, _ = self.sample(t)
        for begin, duration, a, b in self.segments:
            if begin+duration >= t:
                if not world.edge_free(current, b, body):
                    return False
                current = b
        return not world.collision(current, body)


def scene(name):
    """A wall spans the map; the only passage is the deliberately sized aperture."""
    names = ('open', 'narrow', 'low', 'compound', 'blocked')
    if name not in names:
        raise ValueError('Unknown scene: '+name)
    width = {'narrow': .46, 'compound': .46, 'blocked': .30}.get(name, 1.2)
    ceiling = 1.22 if name in ('low', 'compound') else 2.
    boxes = []
    if name != 'open':
        boxes = [(-.3, -2, 0, .3, -width/2, 2), (-.3, width/2, 0, .3, 2, 2)]
        if ceiling < 2:
            boxes.append((-.3, -width/2, ceiling, .3, width/2, 2))
    return VoxelMap(obstacles=boxes), State(-1.2, 0, 0, .85), State(1.2, 0, 0, .85)
