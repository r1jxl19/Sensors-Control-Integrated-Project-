# Development log: what failed, how it was diagnosed, what changed

This is the record behind the viva question *"What failed during development,
how was it diagnosed, and what would you change?"*. Each entry names the
component at fault (model, sensing, estimator, controller or integration).

## 1. Robot crawled at 60 % of commanded speed (model / simulation)

* **Symptom.** With 5 rad/s commanded on both wheels, the wheels settled at
  about 3 rad/s. The body pitched back and forth, and the wheel rates oscillated
  by ±0.3 rad/s.
* **Diagnosis.** A breakdown of generalised forces (`qfrc_actuator`,
  `qfrc_constraint`) showed the actuator torque being cancelled by contact
  forces. The listed contacts showed the ball caster in contact with the
  combined floor friction (μ = 1, condim 3). MuJoCo takes the *maximum*
  friction and condim of the two geoms, so the "frictionless" caster was
  scrubbing on the floor.
* **Fix.** `priority="1"` on the caster, so its own frictionless contact model
  wins. Wheel collision geoms were also changed to spheres, which give a single
  contact point (cylinder–plane gives four rim contacts).
* **Lesson.** Check that the platform model behaves like the datasheet before
  identifying anything from it.

## 2. EKF looked 4–5° wrong in heading during turns (integration / evaluation)

* **Symptom.** The heading error was 0.08 rad during in-place rotation, yet the
  odometry-only heading error over a full in-place turn was only 0.07 %.
* **Diagnosis.** The evaluation logger ran *after* `sim.step()`. Ground truth
  was therefore 50 ms ahead of the estimate, and at 1.2 rad/s that is
  0.06 rad. The "error" was a timestamp mismatch, not an estimator fault.
* **Fix.** The logger is now called just before actuation, so estimate and
  truth refer to the same instant. Mean NEES dropped from 99 to about 70 from
  this change alone.

## 3. Filter overconfident, NEES around 70 (sensing)

* **Symptom.** After fix 2, position errors of about 1 cm persisted with a
  reported σ of 3–5 mm.
* **Diagnosis.** Measurement residuals against truth were logged per phase.
  Bearings were unbiased (σ ≈ 0.1°). Ranges were biased: +4 cm mean with heavy
  positive tails at 2–3.5 m, and −7 mm when docked after a global scale
  correction. Binned by range and view angle, the bias grows roughly as r². That
  is the signature of a near-constant ~0.3 px inward shift of the detected
  corners (texture filtering). A single scale factor fixed long range and broke
  short range.
* **Fix.** A quadratic bias model `c1·rn + c2·rn²` with range normalised to
  marker size, and a size-consistent noise model. Measurements beyond 2.5 m
  (normalised) or viewed at more than 60° are rejected instead of modelled
  (`scripts/characterise_markers.py`). NEES fell to roughly 5–14.

## 4. One trial in 16 docked 32 cm off with a "successful" mission (estimator)

* **Symptom.** In seed 102 the mission ran to DONE, but the robot was 32 cm from
  the dock.
* **Diagnosis.** Estimate error and σ were printed against time. The EKF was
  initialised from the first single-marker PnP pose, and that pose was 24 cm
  off. With σ = 5 cm, every later measurement exceeded the χ² gate: 118 gated,
  0 accepted. The filter coasted on odometry for the whole mission and never
  noticed.
* **Fix.**
  1. *Batch spin localisation* (`tb3dock/localisation.py`). Observations are
     collected over a full revolution and the pose is solved by robust least
     squares over all markers in view. The observability analysis shows that
     two or more markers make the pose observable, whereas one marker leaves a
     rotation-about-the-marker direction unobservable.
  2. *Divergence watchdog.* If 15 consecutive camera frames contain in-envelope
     markers and all are gated, the robot stops and re-localises.
* **Lesson.** A chi-square gate protects the filter from bad measurements but
  also protects a bad estimate from good measurements. Something has to watch
  the gate.

## 5. Odometry noise estimate varied by 100× depending on the statistic (sensing)

* **Symptom.** The mean-squared 1 s increment error gave k ≈ 1e-3 m. A median-based
  estimate gave k ≈ 6e-6 m.
* **Diagnosis.** Errors are heavy-tailed. During hard acceleration the chassis
  pitches, and the encoders (which measure wheel rotation *relative to the
  body*) record motion that does not happen. These errors cancel over longer
  windows; heading k falls as the window grows.
* **Decision.** Use the robust estimate as the starting point, then scale the
  process noise by α chosen from the NEES sweep (`scripts/evaluate.py --exp
  sweep_alpha`). This is reported rather than hidden.

## 6. Combined disturbance, seed 106: replan livelock (integration) - fixed in v2

* **Symptom.** In the final evaluation, 1 of 12 "combined" trials timed out
  with the estimate 1.9 m off.
* **Diagnosis.** `scripts/plot_trial.py --seed 106 --dist combined` (see
  `results/trial_combined_fail_106.png`) shows the sequence:
  1. A push on a low-friction floor spun the robot 45° while the wheels slipped.
  2. The heading error jumped outside the 3σ band.
  3. The LiDAR map was built on the rotated pose; the walls appear skewed in
     the figure.
  4. Every new path looked blocked, so the robot replanned every scan
     (60 replans) while facing a wall with no markers in view.
  5. The watchdog needs in-envelope markers, so it never fired.
* **Fix.** `MissionParams(livelock_relocalise=True)`: eight or more replans
  within 10 s means the map disagrees with the pose, so the robot discards the
  map and re-localises. On the same 24 seeds (push + combined), combined went
  from 11/12 to 12/12, and all other runs were bit-identical. The fix defaults
  to off, so every earlier table is reproducible from the code that produced
  it.

## 7. Blind docking after an unseen push (control / docking): fixed with a dock guard (v3)

* **Symptom.** While testing the LQI extension in the full pipeline, one push
  trial (seed 109) "docked" 33 cm off.
* **Diagnosis.** Estimate error over time showed a push at 31.3 s rotating the
  robot 23° without odometry noticing, 2 s before docking. `FACE_DOCK` aligned
  to the wrong estimated heading, so the dock marker was never detected, and
  the 0.55 m approach was driven blind. The control law was not at fault:
  under LQR the same push landed at a different moment.
* **Fix.** `MissionParams(dock_guard=True)`: approach only while the dock
  marker is detected; if it is unseen for 1.5 s, re-localise. On 48 paired
  trials it fixed this case (LQI push 11/12 → 12/12) and never fired falsely.

## What we would change

* Fuse the gyro by default. It removes wheel-slip heading error, which is the
  main failure mode under low friction and pushes (see the gyro results).
* Estimate the range bias online (augment the state) instead of calibrating it
  once. The bias depends on rendering and lighting, so it will differ on real
  hardware.
* Replace time-indexed reference tracking with path-parameter tracking, so that
  large disturbances do not have to be absorbed by the "reference waits"
  heuristic.
