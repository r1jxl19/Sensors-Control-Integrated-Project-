"""ArUco landmark detection and measurement extraction.

For every decoded marker that belongs to the known map we compute
  * the marker pose in the camera frame (solvePnP, IPPE_SQUARE),
  * the marker centre in the base frame (via T_B_C),
  * the planar range and bearing from the base origin to the marker centre,
    which is the measurement fused by the EKF,
  * diagnostics: pixel area, reprojection error, viewing angle.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from tb3dock.config import WORLD
from .frames import camera_to_base, R_B_C


@dataclass
class MarkerObs:
    id: int
    corners: np.ndarray      # (4, 2) pixels
    rvec: np.ndarray
    tvec: np.ndarray         # marker centre in {C}, m
    p_B: np.ndarray          # marker centre in {B}, m (3,)
    range: float             # planar range in {B}, m
    bearing: float           # rad, CCW from +x_B
    reproj_err: float        # px RMS
    view_angle: float        # rad, angle between marker normal and line of sight
    yaw_B: float             # marker normal direction (yaw) expressed in {B}
    pixel_size: float        # mean side length, px


class MarkerDetector:
    def __init__(self, K, dist, marker_map):
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.map = marker_map
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, WORLD.aruco_dict))
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 4
        self.det = cv2.aruco.ArucoDetector(d, p)

    def detect(self, rgb, min_pixel_size=14.0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i not in self.map:
                continue              # unknown id -> ignore (could be a false decode)
            c = c.reshape(4, 2).astype(np.float64)
            side = np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1))
            if side < min_pixel_size:
                continue              # too small for a reliable pose
            s = self.map[i]["size"] / 2
            obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - c) ** 2, 1))))
            t = tvec.ravel()
            p_B = camera_to_base(t)
            R_C_K, _ = cv2.Rodrigues(rvec)
            n_B = R_B_C @ R_C_K[:, 2]            # marker z axis (out of wall) in {B}
            los = t / np.linalg.norm(t)
            view = float(np.arccos(np.clip(-los @ R_C_K[:, 2], -1, 1)))
            out.append(MarkerObs(
                id=i, corners=c, rvec=rvec.ravel(), tvec=t, p_B=p_B,
                range=float(np.hypot(p_B[0], p_B[1])),
                bearing=float(np.arctan2(p_B[1], p_B[0])),
                reproj_err=err, view_angle=view,
                yaw_B=float(np.arctan2(n_B[1], n_B[0])), pixel_size=float(side)))
        return out
