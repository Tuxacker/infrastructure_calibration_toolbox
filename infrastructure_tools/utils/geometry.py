import numpy as np
from numba import njit

@njit(fastmath=True)
def cross_prod_2d(x, y):
    return x[0] * y[1] - x[1] * y[0]


@njit(fastmath=True)
def line_intersection2d(p1, p2, q1, q2):
    p = p1
    r = p2 - p1
    q = q1
    s = q2 - q1
    if np.abs(cross_prod_2d(r, s)) < 1e-6:
        return p * np.nan  # np.array([np.nan, np.nan], dtype=np.float64)
    t = cross_prod_2d(q - p, s) / cross_prod_2d(r, s)
    return p + t * r


@njit(fastmath=True)
def triangle_area(x, y, z):
    return 0.5 * np.abs((y[0] * x[1] - x[0] * y[1]) + (z[0] * y[1] - y[0] * z[1]) + (x[0] * z[1] - z[0] * x[1]))


@njit(fastmath=True)
def point_rect_intersection(p, rect):
    a, b, c, d = rect
    eps = 1e-3

    rect_area = triangle_area(a, b, c) + triangle_area(c, d, a)
    point_area = triangle_area(a, b, p) + triangle_area(b, c, p) + \
                 triangle_area(c, d, p) + triangle_area(d, a, p)

    if point_area > rect_area + eps:
        return False
    else:
        return True


def distance_line_to_point(p, p1, p2):
    den = np.linalg.norm(p2 - p1)
    num = np.abs((p2[0] - p1[0]) * (p1[1] - p[1]) - (p1[0] - p[0]) * (p2[1] - p1[1]))
    return num / den


def dist_to_gp(p, gp):
    dists = []
    for i in range(4):
        dists.append(distance_line_to_point(p, gp[i], gp[(i + 1) % 4]))
    return np.array(dists).min()


def sort_polygon(np_poly):
    assert type(np_poly) == np.ndarray and np_poly.shape[-1] == 2 and np_poly.ndim == 2
    middle = np.mean(np_poly, axis=0)
    np_poly_local = np.copy(np_poly) - middle
    angles = np.arctan2(np_poly_local[:, 1], np_poly_local[:, 0])
    indices = np.argsort(angles)
    return np_poly[indices]


@njit(fastmath=True)
def normalize_angle(a, period=2 * np.pi):
    return (a + period / 2.0) % period - period / 2.0


@njit(fastmath=True)
def normalize(v):
    return v / np.linalg.norm(v)


def to_homogenous_single(v):
    assert v.ndim == 1
    return np.array([*v, 1])


def to_homogenous_batch(v):
    if v.ndim == 1:
        v = v.reshape((1, -1))
    return np.c_[v, np.ones(len(v))]


def vec2angle(_dir):
    return np.arctan2(_dir[1], _dir[0])


def get_intersection3d(o, e, q1, q2):
    X = np.zeros((3, 2))
    X[:, 0] = e - o
    X[:, 1] = q1 - q2
    y = q1 - o
    p = np.linalg.inv(X.T @ X) @ X.T @ y
    if np.linalg.norm(X @ p - y) > 1e-6:
        return np.array([np.nan, np.nan, np.nan])
    return o + p[0] * (e - o)


def vehicle_volume_from_imu(imu_state: np.ndarray, imu_offset: np.ndarray, vehicle_length: float, vehicle_width: float, vehicle_height: float):
    # TODO: Fix CARLA output state

    if len(imu_state) == 4:
        x, y, z, yaw, = imu_state
        # pitch = roll = 0.0
    else:
        x, y, z, yaw, _, _ = imu_state # pitch, roll are discarded

    corners = np.array([
        [0, 0, 0],
        [0, 0, vehicle_height],
        [0, vehicle_width, vehicle_height],
        [0, vehicle_width, 0],
        [vehicle_length, vehicle_width, 0],
        [vehicle_length, vehicle_width, vehicle_height],
        [vehicle_length, 0, vehicle_height],
        [vehicle_length, 0, 0]
    ])
    corners -= imu_offset

    sy, cy = np.sin(yaw), np.cos(yaw)

    # Pitch and roll are currently discarded

    # sp, cp = np.sin(-pitch), np.cos(-pitch)  # yaw and pitch are negative in ADMA corrdinate system
    # sr, cr = np.sin(roll), np.cos(roll)

    # rotation_matrix = np.array([
    #     [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
    #     [sy*cp, sy*sp*sr-cy*cr, sy*sp*cr-cy*sr],
    #     [-sp, cp*sr, cp*cr]
    # ])

    rotation_matrix = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return (rotation_matrix @ corners.T).T + np.array([[x, y, z]])


def vehicle_polygon_from_imu(imu_state: np.ndarray, imu_offset: np.ndarray, vehicle_length: float, vehicle_width: float):
    
    if len(imu_state) == 4:
        x, y, _, yaw = imu_state
    else:
        x, y, _, yaw, _, _ = imu_state
    corners = np.array([
        [0, 0],
        [0, vehicle_width],
        [vehicle_length, vehicle_width],
        [vehicle_length, 0]
    ])
    corners -= imu_offset

    sy, cy = np.sin(yaw), np.cos(yaw)

    rotation_matrix = np.array([
        [cy, -sy],
        [sy, cy]
    ])

    return (rotation_matrix @ corners.T).T + np.array([[x, y]])


def vehicle_polygon_from_etsi_cam(cam: np.ndarray):
    x, y, yaw, l, w, _ = cam # station_id is discarded
    corners = np.array([
        [0, 0],
        [0, w],
        [l, w],
        [l, 0]
    ])
    corners -= np.array([[l / 2, w / 2]])

    sy, cy = np.sin(yaw), np.cos(yaw)

    rotation_matrix = np.array([
        [cy, -sy],
        [sy, cy]
    ])

    return (rotation_matrix @ corners.T).T + np.array([[x, y]])