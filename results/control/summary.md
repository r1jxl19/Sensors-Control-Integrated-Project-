# Part D control evaluation (standalone plant, 40 paired random paths)

## A. Cross-track error vs ground truth, RMS per path (median over paths, mm), and p95 of max

| Condition | LQR median | LQI median | Kanayama median | LQR p95 max | LQI p95 max | Kanayama p95 max |
|---|---|---|---|---|---|---|
| perfect estimate | 4.8 | 3.8 | 4.7 | 26.9 | 25.7 | 28.7 |
| nominal | 10.1 | 10.6 | 10.0 | 35.3 | 34.9 | 35.1 |
| 3x estimate noise | 26.7 | 27.3 | 26.3 | 91.2 | 90.7 | 89.1 |
| motor mismatch | 14.3 | 11.0 | 14.3 | 41.6 | 41.5 | 45.3 |
| slow actuator (60 ms) | 10.2 | 10.8 | 10.0 | 36.0 | 35.9 | 35.9 |
| 50 ms delay | 10.2 | 10.9 | 10.0 | 36.5 | 36.3 | 36.4 |
| pushes | 20.8 | 19.8 | 20.8 | 84.2 | 84.9 | 83.5 |

## A. Other metrics (median over paths)

| Condition | Controller | heading RMS (deg) | effort RMS dω/dt (rad/s²) | saturated (% of steps) | final position error (mm) | finish − nominal (s) |
|---|---|---|---|---|---|---|
| perfect estimate | LQR | 1.19 | 0.53 | 10.8 | 3.3 | -0.03 |
| perfect estimate | LQI | 1.31 | 0.53 | 10.9 | 3.4 | -0.03 |
| perfect estimate | Kanayama | 1.32 | 0.52 | 10.6 | 3.3 | -0.03 |
| nominal | LQR | 1.91 | 1.10 | 10.6 | 12.3 | -0.03 |
| nominal | LQI | 2.37 | 1.43 | 11.7 | 13.3 | -0.03 |
| nominal | Kanayama | 2.00 | 1.08 | 10.6 | 12.4 | -0.03 |
| 3x estimate noise | LQR | 4.37 | 2.81 | 44.3 | 38.5 | -0.03 |
| 3x estimate noise | LQI | 5.36 | 3.73 | 52.4 | 40.4 | -0.03 |
| 3x estimate noise | Kanayama | 4.40 | 2.77 | 44.8 | 38.3 | -0.03 |
| motor mismatch | LQR | 2.06 | 1.07 | 24.6 | 16.9 | -0.03 |
| motor mismatch | LQI | 2.36 | 1.41 | 26.9 | 13.6 | -0.03 |
| motor mismatch | Kanayama | 2.12 | 1.05 | 23.7 | 16.7 | -0.03 |
| slow actuator (60 ms) | LQR | 1.97 | 1.10 | 10.7 | 14.1 | -0.03 |
| slow actuator (60 ms) | LQI | 2.44 | 1.44 | 12.6 | 15.1 | -0.03 |
| slow actuator (60 ms) | Kanayama | 2.07 | 1.09 | 11.1 | 14.3 | -0.03 |
| 50 ms delay | LQR | 2.00 | 1.14 | 11.2 | 14.8 | -0.03 |
| 50 ms delay | LQI | 2.49 | 1.49 | 13.4 | 15.5 | -0.03 |
| 50 ms delay | Kanayama | 2.10 | 1.12 | 11.5 | 15.0 | -0.03 |
| pushes | LQR | 3.45 | 1.41 | 20.0 | 12.4 | -0.03 |
| pushes | LQI | 4.04 | 1.72 | 22.4 | 13.3 | -0.03 |
| pushes | Kanayama | 3.56 | 1.42 | 20.0 | 12.7 | -0.03 |

## A. Paired differences vs LQR (same path, same noise): cross-track RMS

| Condition | LQI better on | LQI median change (mm) | Kanayama better on | Kanayama median change (mm) |
|---|---|---|---|---|
| perfect estimate | 40/40 | -0.8 | 15/40 | 0.1 |
| nominal | 8/40 | 0.9 | 18/40 | 0.0 |
| 3x estimate noise | 9/40 | 1.2 | 29/40 | -0.3 |
| motor mismatch | 38/40 | -3.8 | 26/40 | -0.1 |
| slow actuator (60 ms) | 9/40 | 0.8 | 19/40 | 0.0 |
| 50 ms delay | 9/40 | 0.7 | 19/40 | 0.0 |
| pushes | 34/40 | -1.1 | 24/40 | -0.1 |

## B. LQR input-weight sweep (median over paths)

| Condition | R scale | cross-track vs truth (mm) | cross-track vs estimate (mm) | effort (rad/s²) | saturated (%) |
|---|---|---|---|---|---|
| nominal | 0.1 | 10.6 | 10.9 | 2.56 | 23.3 |
| nominal | 0.3 | 9.9 | 10.7 | 1.75 | 14.8 |
| nominal | 1 | 10.1 | 10.5 | 1.10 | 10.6 |
| nominal | 3 | 9.5 | 10.2 | 0.74 | 9.8 |
| nominal | 10 | 9.2 | 10.3 | 0.50 | 9.4 |
| 3x estimate noise | 0.1 | 27.1 | 28.9 | 6.14 | 72.9 |
| 3x estimate noise | 0.3 | 27.0 | 28.2 | 4.41 | 60.0 |
| 3x estimate noise | 1 | 26.7 | 28.1 | 2.81 | 44.3 |
| 3x estimate noise | 3 | 25.7 | 28.3 | 1.79 | 28.8 |
| 3x estimate noise | 10 | 25.2 | 29.6 | 1.08 | 15.2 |

## C. Docking approach: final cross-track error at the dock (mm)

| Condition | Controller | median abs | p95 abs | max abs |
|---|---|---|---|---|
| nominal | LQR | 4.3 | 11.6 | 14.1 |
| nominal | LQI | 5.5 | 20.4 | 23.2 |
| nominal | Kanayama | 4.8 | 11.6 | 13.6 |
| motor mismatch | LQR | 5.3 | 14.2 | 17.4 |
| motor mismatch | LQI | 4.7 | 21.5 | 23.0 |
| motor mismatch | Kanayama | 10.2 | 21.1 | 23.6 |
