"""tb3sensing - sensing, calibration and frame management (Track A, Part B).

Everything between the raw sensors and the estimator:

    frames       frame definitions {M} {B} {C} {L} {K} and transforms
    vision       ArUco detection, PnP, range/bearing in the base frame
    measurement  marker measurement model: envelope, range-bias correction, R
    odometry     wheel-encoder odometry (ticks -> arc lengths -> pose increment)
    lidar        LiDAR validity, beam -> point transforms, map-point error budget

Calibration and characterisation scripts (they write config/calibration.json):
    scripts/calibrate_camera.py      intrinsics K and distortion
    scripts/calibrate_odometry.py    wheel radii, track, odometry noise k
    scripts/characterise_markers.py  range bias, R, envelope
    scripts/sensing_lidar.py         LiDAR noise, bias and dropout
    scripts/sensing_frames.py        frame-chain check and extrinsic sensitivity
See tb3sensing/README.md for the Part B write-up.
"""
