"""LiDAR occupancy grid (log-odds) built on the *estimated* pose.

The robot is told only the room footprint; walls, obstacles and the dock are
discovered from LiDAR. Unknown space is treated as free by the planner
(optimistic planning) and the path is re-checked after every scan.
"""
import numpy as np
from scipy.ndimage import binary_dilation

from .config import WORLD, LIDAR
from .frames import t_B_L


class OccupancyGrid:
    L_OCC, L_FREE, L_MIN, L_MAX, L_THRESH = 0.85, -0.4, -2.0, 3.5, 0.6

    def __init__(self, res=0.05, margin=0.3):
        self.res = res
        self.origin = np.array([-margin, -margin])
        self.nx = int(np.ceil((WORLD.size_x + 2 * margin) / res))
        self.ny = int(np.ceil((WORLD.size_y + 2 * margin) / res))
        self.logodds = np.zeros((self.nx, self.ny))
        self._inflated = None
        self._infl_r = None

    def world_to_cell(self, p):
        c = np.floor((np.asarray(p)[..., :2] - self.origin) / self.res).astype(int)
        return c

    def cell_to_world(self, c):
        return self.origin + (np.asarray(c) + 0.5) * self.res

    def in_bounds(self, c):
        c = np.asarray(c)
        return (c[..., 0] >= 0) & (c[..., 0] < self.nx) & (c[..., 1] >= 0) & (c[..., 1] < self.ny)

    def update(self, pose, ranges, angles, max_range=LIDAR.r_max):
        """Integrate one scan taken at estimated pose (x, y, theta)."""
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        o = np.array([x + c * t_B_L[0] - s * t_B_L[1], y + s * t_B_L[0] + c * t_B_L[1]])
        oc = self.world_to_cell(o)
        hit = ~np.isnan(ranges)
        r = np.where(hit, ranges, max_range)
        a = th + angles
        ends = o + r[:, None] * np.stack([np.cos(a), np.sin(a)], 1)
        step = self.res * 0.7
        d = np.arange(0, r.max(), step)                       # distances along every beam
        pts = o + d[None, :, None] * np.stack([np.cos(a), np.sin(a)], 1)[:, None, :]
        mask = d[None, :] < (r[:, None] - self.res)
        fc = self.world_to_cell(pts[mask])
        if len(fc):
            fc = np.unique(fc, axis=0)
            fc = fc[self.in_bounds(fc)]
            self.logodds[fc[:, 0], fc[:, 1]] += self.L_FREE
        hc = self.world_to_cell(ends[hit])
        hc = hc[self.in_bounds(hc)]
        self.logodds[hc[:, 0], hc[:, 1]] += self.L_OCC
        np.clip(self.logodds, self.L_MIN, self.L_MAX, out=self.logodds)
        self._inflated = None
        return oc

    def occupied(self):
        return self.logodds > self.L_THRESH

    def inflated(self, radius):
        if self._inflated is None or self._infl_r != radius:
            k = int(np.ceil(radius / self.res))
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            disk = (xx ** 2 + yy ** 2) <= (radius / self.res) ** 2
            occ = self.occupied()
            # room boundary is known a priori
            ix0, iy0 = self.world_to_cell([0, 0])
            ix1, iy1 = self.world_to_cell([WORLD.size_x, WORLD.size_y])
            occ = occ.copy()
            occ[:ix0 + 1, :] = True
            occ[ix1:, :] = True
            occ[:, :iy0 + 1] = True
            occ[:, iy1:] = True
            self._inflated = binary_dilation(occ, structure=disk)
            self._infl_r = radius
        return self._inflated
