"""Path planning and trajectory generation.

A* on the inflated occupancy grid (8-connected, with a clearance penalty from
a distance transform) -> line-of-sight shortcutting -> corner rounding ->
time-parameterised reference (x_r, y_r, theta_r, v_r, omega_r) respecting
speed, lateral-acceleration, yaw-rate and longitudinal-acceleration limits.
"""
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt

from .frames import wrap


def astar(blocked, start, goal, clearance=None, w_clear=2.0):
    nx, ny = blocked.shape
    start, goal = tuple(start), tuple(goal)
    if blocked[start] or blocked[goal]:
        return None
    nbrs = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    g = {start: 0.0}
    came = {}
    h = lambda c: np.hypot(c[0] - goal[0], c[1] - goal[1])
    openq = [(h(start), start)]
    closed = set()
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dx, dy, c in nbrs:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or blocked[n]:
                continue
            pen = 0.0 if clearance is None else w_clear / (1.0 + clearance[n])
            ng = g[cur] + c + pen
            if ng < g.get(n, np.inf):
                g[n] = ng
                came[n] = cur
                heapq.heappush(openq, (ng + h(n), n))
    return None


def line_free(blocked, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    xs = np.round(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.round(np.linspace(a[1], b[1], n)).astype(int)
    return not blocked[xs, ys].any()


def shortcut(blocked, path):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def nearest_free(blocked, c, max_r=12):
    c = np.asarray(c)
    if not blocked[tuple(c)]:
        return tuple(c)
    for r in range(1, max_r):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                n = c + [dx, dy]
                if 0 <= n[0] < blocked.shape[0] and 0 <= n[1] < blocked.shape[1] and not blocked[tuple(n)]:
                    return tuple(n)
    return None


def plan_path(grid, start_xy, goal_xy, robot_radius):
    blocked = grid.inflated(robot_radius)
    s = nearest_free(blocked, grid.world_to_cell(start_xy))
    g = nearest_free(blocked, grid.world_to_cell(goal_xy))
    if s is None or g is None:
        return None
    clearance = distance_transform_edt(~blocked)
    cells = astar(blocked, s, g, clearance)
    if cells is None:
        return None
    cells = shortcut(blocked, cells)
    pts = np.array([grid.cell_to_world(c) for c in cells])
    pts[0] = start_xy
    pts[-1] = goal_xy
    return pts


def round_corners(pts, ds=0.02, radius=0.25):
    """Resample a polyline and round its corners with circular-ish blends."""
    if len(pts) < 3:
        return resample(pts, ds)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
        r = min(radius, 0.45 * la, 0.45 * lc)
        p0 = b + (a - b) / la * r
        p2 = b + (c - b) / lc * r
        t = np.linspace(0, 1, 12)[:, None]
        bez = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2   # quadratic Bezier blend
        out.extend(bez)
    out.append(pts[-1])
    return resample(np.array(out), ds)


def resample(pts, ds):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    pts = pts[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] < 1e-6:
        return pts[:1]
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


# Reference generation is part of the control package (tb3control.reference).
from tb3control.reference import Trajectory  # noqa: E402,F401
