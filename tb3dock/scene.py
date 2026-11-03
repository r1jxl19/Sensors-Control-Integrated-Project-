"""MJCF scene builder for the TurtleBot3 docking task.

Builds a 5 m x 4 m room containing:
  * a TurtleBot3 Burger-like differential-drive robot (wheels, ball caster,
    LDS LiDAR housing, forward camera) using the hidden TRUE geometry;
  * randomisable box/cylinder obstacles (unknown to the robot);
  * eight ArUco landmarks on the walls (known map, used by the EKF);
  * a charging dock on the +x wall carrying ArUco id 0.

Map frame {M}: origin at the room corner, x along the 5 m wall, z up.
"""
import os
import numpy as np
import cv2

from .config import ASSET_DIR, TRUE, CAMERA, LIDAR, WORLD

# Geom groups: 0 = environment (seen by LiDAR), 2 = robot, 1 = marker decals (visual only).
G_ENV, G_ROBOT, G_DECAL = 0, 2, 1
MARKER_BORDER = 0.25   # white quiet zone on each side, as fraction of marker edge
CHECKER_SQUARE = 0.03  # m, calibration board square edge


def _aruco_dict():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, WORLD.aruco_dict))


def make_marker_textures():
    """Write ArUco marker PNGs (with white quiet zone) into assets/."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    d = _aruco_dict()
    ids = [WORLD.dock_marker_id] + [l[0] for l in WORLD.landmarks]
    px = 240
    pad = int(px * MARKER_BORDER)
    for i in ids:
        path = os.path.join(ASSET_DIR, f"aruco_{i}.png")
        if os.path.exists(path):
            continue
        img = cv2.aruco.generateImageMarker(d, i, px)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    # checkerboard for camera calibration (9x6 inner corners)
    path = os.path.join(ASSET_DIR, "checker.png")
    if not os.path.exists(path):
        sq, nx, ny, m = 60, 10, 7, 60
        img = np.full((ny * sq + 2 * m, nx * sq + 2 * m), 255, np.uint8)
        for r in range(ny):
            for c in range(nx):
                if (r + c) % 2 == 0:
                    img[m + r * sq:m + (r + 1) * sq, m + c * sq:m + (c + 1) * sq] = 0
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


def _wall_pose(wall, s, eps=0.002):
    """Position on an inner wall face and inward normal."""
    X, Y = WORLD.size_x, WORLD.size_y
    if wall == "y0":
        return np.array([s, eps]), np.array([0.0, 1.0])
    if wall == "y4":
        return np.array([s, Y - eps]), np.array([0.0, -1.0])
    if wall == "x0":
        return np.array([eps, s]), np.array([1.0, 0.0])
    if wall == "x5":
        return np.array([X - eps, s]), np.array([-1.0, 0.0])
    raise ValueError(wall)


def marker_map():
    """Known landmark map: id -> centre position (3,), inward normal (3,), size.

    This is the only world knowledge the robot is given (plus room size).
    """
    m = {}
    for mid, wall, s in WORLD.landmarks:
        p, n = _wall_pose(wall, s)
        m[mid] = dict(p=[p[0], p[1], WORLD.landmark_h], n=[n[0], n[1], 0.0],
                      size=WORLD.landmark_size)
    px = WORLD.size_x - WORLD.dock_block_depth - 0.001
    m[WORLD.dock_marker_id] = dict(p=[px, WORLD.dock_y, WORLD.dock_marker_h],
                                   n=[-1.0, 0.0, 0.0], size=WORLD.dock_marker_size)
    return m


def _decal_xml(mid, p, n, size):
    """Textured quad facing along n, upright, black square of edge `size`."""
    n = np.asarray(n, float)
    right = np.cross(-n, [0, 0, 1.0])          # viewer's right when facing the decal
    full = size * (1 + 2 * MARKER_BORDER)
    off = p + n * 0.0005
    return (f'<geom name="decal_{mid}" type="mesh" mesh="quad" material="mat_aruco_{mid}" '
            f'pos="{off[0]:.4f} {off[1]:.4f} {off[2]:.4f}" '
            f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} 0 0 1" '
            f'contype="0" conaffinity="0" group="{G_DECAL}" '
            f'mass="0" />'), full


def random_scenario(rng, n_obstacles=None, max_tries=200):
    """Random start pose and obstacle layout.

    Obstacles are kept out of the start area and the dock approach corridor,
    with enough spacing that a path normally exists (the planner still has
    to find it from LiDAR alone).
    """
    from .config import dock_goal
    if n_obstacles is None:
        n_obstacles = int(rng.integers(3, 6))
    start = np.array([rng.uniform(0.45, 1.0), rng.uniform(0.6, 3.4),
                      rng.uniform(-np.pi, np.pi)])
    goal = np.array(dock_goal())
    obs = []
    tries = 0
    while len(obs) < n_obstacles and tries < max_tries:
        tries += 1
        kind = "box" if rng.random() < 0.6 else "cylinder"
        c = np.array([rng.uniform(1.4, 3.9), rng.uniform(0.6, 3.4)])
        if kind == "box":
            half = rng.uniform(0.10, 0.22, size=2)
            yaw = rng.uniform(0, np.pi)
            rad = float(np.hypot(*half))
        else:
            half = np.array([rng.uniform(0.08, 0.16)] * 2)
            yaw = 0.0
            rad = float(half[0])
        if np.linalg.norm(c - start[:2]) < rad + 0.55:
            continue
        if c[0] + rad > 3.55 and abs(c[1] - goal[1]) < rad + 0.45:
            continue
        if any(np.linalg.norm(c - o["c"]) < rad + o["rad"] + 0.45 for o in obs):
            continue
        obs.append(dict(kind=kind, c=c, half=half, yaw=yaw, rad=rad))
    return dict(start=start, obstacles=obs)


def build_xml(scenario=None):
    make_marker_textures()
    if scenario is None:
        scenario = dict(start=np.array([0.7, 1.0, 0.0]), obstacles=[])
    X, Y, H = WORLD.size_x, WORLD.size_y, WORLD.wall_h
    sx, sy, sth = scenario["start"]
    rl, rr, tr = TRUE.wheel_radius_l, TRUE.wheel_radius_r, TRUE.track
    axle_h = 0.033

    mm = marker_map()
    textures, materials, decals = [], [], []
    for mid, m in mm.items():
        textures.append(f'<texture name="tex_aruco_{mid}" type="2d" file="aruco_{mid}.png"/>')
        materials.append(f'<material name="mat_aruco_{mid}" texture="tex_aruco_{mid}" '
                         f'specular="0" shininess="0" emission="0.25"/>')
        xml, full = _decal_xml(mid, np.array(m["p"]), m["n"], m["size"])
        # mesh is unit quad; scale via a dedicated mesh per size
        decals.append((xml, full))
    sizes = sorted({round(f, 4) for _, f in decals})
    meshes = []
    for s in sizes:
        h = s / 2
        meshes.append(
            f'<mesh name="quad_{s}" vertex="{-h} {-h} 0  {h} {-h} 0  {h} {h} 0  {-h} {h} 0" '
            f'texcoord="0 1  1 1  1 0  0 0" face="0 1 2  0 2 3" inertia="shell"/>')
    decal_xml = "\n".join(x.replace('mesh="quad"', f'mesh="quad_{round(f, 4)}"') for x, f in decals)

    obst = []
    for k, o in enumerate(scenario["obstacles"]):
        cx, cy = o["c"]
        if o["kind"] == "box":
            obst.append(f'<geom name="obs{k}" type="box" size="{o["half"][0]:.3f} {o["half"][1]:.3f} 0.175" '
                        f'pos="{cx:.3f} {cy:.3f} 0.175" euler="0 0 {np.degrees(o["yaw"]):.2f}" '
                        f'material="mat_obs" group="{G_ENV}"/>')
        else:
            obst.append(f'<geom name="obs{k}" type="cylinder" size="{o["half"][0]:.3f} 0.175" '
                        f'pos="{cx:.3f} {cy:.3f} 0.175" material="mat_obs2" group="{G_ENV}"/>')

    dock_x = X - WORLD.dock_block_depth / 2
    dock = (f'<geom name="dock" type="box" size="{WORLD.dock_block_depth / 2} 0.12 0.10" '
            f'pos="{dock_x} {WORLD.dock_y} 0.10" rgba="0.15 0.45 0.25 1" group="{G_ENV}"/>'
            f'<geom name="dock_pad" type="box" size="0.12 0.14 0.002" '
            f'pos="{X - 0.17} {WORLD.dock_y} 0.002" rgba="0.2 0.6 0.3 1" contype="0" conaffinity="0" group="{G_DECAL}"/>')

    board = ""
    if scenario.get("calib_board"):
        sq = CHECKER_SQUARE
        hw, hh = sq * 12 / 2, sq * 9 / 2      # 10x7 squares + 1-square margin each side
        meshes.append(f'<mesh name="checker_quad" vertex="{-hw} {-hh} 0  {hw} {-hh} 0  {hw} {hh} 0  {-hw} {hh} 0" '
                      f'texcoord="0 1  1 1  1 0  0 0" face="0 1 2  0 2 3" inertia="shell"/>')
        board = ('<body name="board" mocap="true" pos="2.5 2 0.3">'
                 '<geom type="mesh" mesh="checker_quad" material="mat_checker" contype="0" conaffinity="0" '
                 f'group="{G_DECAL}" mass="0"/></body>')

    walls = ""
    if not scenario.get("no_walls"):
        for name, sz, pos in (("wall_y0", (X / 2 + 0.05, 0.05), (X / 2, -0.05)),
                              ("wall_y4", (X / 2 + 0.05, 0.05), (X / 2, Y + 0.05)),
                              ("wall_x0", (0.05, Y / 2 + 0.05), (-0.05, Y / 2)),
                              ("wall_x5", (0.05, Y / 2 + 0.05), (X + 0.05, Y / 2))):
            walls += (f'<geom name="{name}" type="box" size="{sz[0]} {sz[1]} {H / 2}" '
                      f'pos="{pos[0]} {pos[1]} {H / 2}" material="mat_wall" group="{G_ENV}"/>\n')

    cam = CAMERA.pos_b
    ldr = LIDAR.pos_b
    xml = f"""
<mujoco model="tb3_docking">
  <compiler angle="degree" texturedir="{ASSET_DIR}" meshdir="{ASSET_DIR}"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="0"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.62 0.6 0.56" rgb2="0.55 0.53 0.5" width="256" height="256"/>
    <material name="mat_floor" texture="floor" texrepeat="10 8"/>
    <material name="mat_wall" rgba="0.85 0.85 0.82 1"/>
    <material name="mat_obs" rgba="0.75 0.35 0.2 1"/>
    <material name="mat_obs2" rgba="0.3 0.4 0.75 1"/>
    <texture name="tex_checker" type="2d" file="checker.png"/>
    <material name="mat_checker" texture="tex_checker" specular="0" emission="0.25"/>
    {''.join(textures)}
    {''.join(materials)}
    {''.join(meshes)}
  </asset>
  <worldbody>
    <light pos="{X / 2} {Y / 2} 3" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="{X} {Y} 0.1" pos="{X / 2} {Y / 2} 0" material="mat_floor" group="{G_ENV}" friction="1 0.005 0.0001"/>
    {walls}
    {dock}
    {decal_xml}
    {''.join(obst)}
    {board}
    <camera name="top" pos="{X / 2} {Y / 2} 7.2" xyaxes="1 0 0 0 1 0" fovy="45"/>

    <body name="base" pos="{sx} {sy} {axle_h}" euler="0 0 {np.degrees(sth):.3f}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.07 0.07 0.05" pos="-0.02 0 0.05" mass="0.47" rgba="0.15 0.15 0.15 1" group="{G_ROBOT}"/>
      <geom name="battery" type="box" size="0.03 0.04 0.012" pos="-0.055 0 0.014" mass="0.35" rgba="0.2 0.2 0.2 1" group="{G_ROBOT}" contype="0" conaffinity="0"/>
      <geom name="plate" type="box" size="0.071 0.071 0.004" pos="-0.02 0 0.1" rgba="0.3 0.3 0.3 1" mass="0.02" group="{G_ROBOT}" contype="0" conaffinity="0"/>
      <geom name="lds" type="cylinder" size="0.035 0.018" pos="{ldr[0]} 0 {ldr[2] - axle_h + 0.01}" rgba="0.2 0.2 0.25 1" mass="0.13" group="{G_ROBOT}" contype="0" conaffinity="0"/>
      <geom name="caster" type="sphere" size="0.012" pos="-0.085 0 -0.021" friction="0 0 0" condim="1" priority="1" mass="0.01" rgba="0.7 0.7 0.7 1" group="{G_ROBOT}"/>
      <geom name="camhousing" type="box" size="0.008 0.015 0.012" pos="{cam[0] - 0.008} 0 {cam[2] - axle_h}" rgba="0.1 0.5 0.1 1" mass="0.01" group="{G_ROBOT}" contype="0" conaffinity="0"/>
      <camera name="cam" pos="{cam[0]} {cam[1]} {cam[2] - axle_h}" xyaxes="0 -1 0 0 0 1" fovy="{CAMERA.fovy_deg}"/>
      <camera name="chase" pos="-0.9 0 0.55" xyaxes="0 -1 0 0.5 0 1" fovy="60"/>
      <site name="imu" pos="0 0 0.02"/>
      <site name="lidar_site" pos="{ldr[0]} {ldr[1]} {ldr[2] - axle_h}"/>
      <body name="wheel_l" pos="0 {tr / 2} {rl - axle_h}">
        <joint name="jl" type="hinge" axis="0 1 0" damping="0.0002"/>
        <geom name="wheel_l_col" type="sphere" size="{rl}" mass="0.03" friction="1.2 0.01 0.001" rgba="0 0 0 0" group="{G_ROBOT}"/>
        <geom type="cylinder" size="{rl} 0.009" euler="90 0 0" mass="0" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" group="{G_ROBOT}"/>
      </body>
      <body name="wheel_r" pos="0 {-tr / 2} {rr - axle_h}">
        <joint name="jr" type="hinge" axis="0 1 0" damping="0.0002"/>
        <geom name="wheel_r_col" type="sphere" size="{rr}" mass="0.03" friction="1.2 0.01 0.001" rgba="0 0 0 0" group="{G_ROBOT}"/>
        <geom type="cylinder" size="{rr} 0.009" euler="90 0 0" mass="0" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" group="{G_ROBOT}"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="ml" joint="jl" kv="0.02" ctrlrange="-7 7" forcerange="-1.2 1.2"/>
    <velocity name="mr" joint="jr" kv="0.02" ctrlrange="-7 7" forcerange="-1.2 1.2"/>
  </actuator>
  <sensor>
    <jointpos name="enc_l" joint="jl"/>
    <jointpos name="enc_r" joint="jr"/>
    <gyro name="gyro" site="imu"/>
  </sensor>
</mujoco>
"""
    return xml
