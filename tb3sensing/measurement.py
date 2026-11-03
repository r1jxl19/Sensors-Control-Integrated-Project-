"""Marker measurement model: the sensing side of every EKF update.

Turns a raw ArUco detection (range r and bearing b in the base frame {B},
vision.MarkerDetector) into a measurement the estimator can trust:

  1. ENVELOPE  reject detections outside the region the model was fitted on:
               normalised range rn > 2.5 m or viewing angle > 60 deg. Beyond
               that the errors become heavy-tailed (grazing views, few pixels
               per marker cell) and a Gaussian R would lie.
  2. BIAS      PnP reads range long by an amount growing ~ rn^2 (a near-constant
               ~0.3 px inward shift of the detected corners). Corrected with
                   r_corr = r - (c1 rn + c2 rn^2) * size / 0.16
  3. NOISE     R = beta * diag(sigma_r^2, sigma_b^2),
                   sigma_r = (a + b rn^2) * size / 0.16,   sigma_b constant
               fitted from 647 detections at surveyed poses, inflated 1.5x.

rn = r * 0.16 / size is range normalised to a 160 mm marker: PnP error scales
with how many pixels the marker covers, so one model serves the 160 mm wall
landmarks and the 100 mm dock marker.

Coefficients come from scripts/characterise_markers.py via
config/calibration.json (range_bias_c, meas_sigma_r, meas_sigma_b,
meas_max_range_norm).
"""
import numpy as np

S0 = 0.16                  # m, reference marker size (wall landmarks)
MAX_VIEW_DEG = 60.0        # deg, envelope on viewing angle


class MarkerMeasurementModel:
    def __init__(self, cal, beta=1.0, max_view_deg=MAX_VIEW_DEG):
        self.bias_c = cal.get("range_bias_c", [0.0, 0.0])
        self.sr_a, self.sr_b = cal["meas_sigma_r"]
        self.sb = cal["meas_sigma_b"]
        self.r_max = cal.get("meas_max_range_norm", 3.0)
        self.beta = beta
        self.max_view = np.radians(max_view_deg)

    @staticmethod
    def normalised_range(r, size):
        return r * S0 / size

    def in_envelope(self, r, size, view_angle):
        return not (r * S0 / size > self.r_max or view_angle > self.max_view)

    def correct_range(self, r, size):
        rn = r * S0 / size
        return r - (self.bias_c[0] * rn + self.bias_c[1] * rn ** 2) * size / S0

    def sigma_range(self, r, size):
        rn = r * S0 / size
        return (self.sr_a + self.sr_b * rn ** 2) * size / S0

    def covariance(self, r, size):
        sr = self.sigma_range(r, size)
        return self.beta * np.diag([sr ** 2, self.sb ** 2])
