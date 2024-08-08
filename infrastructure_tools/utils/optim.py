from math import pi, sin, cos
import os

import cv2
import numpy as np
from scipy.optimize import minimize

from .enums import Classification

if "ENABLE_CALIBRATION" in os.environ:
    # try:
    from excalibur.calibration.point2point import MatrixQCQP as P2P
    from excalibur.calibration.point2line import MatrixQCQP as P2L
    from excalibur.calibration.point2plane import MatrixQCQP as P2PL
    from excalibur.fitting.line import Line
    from excalibur.fitting.plane import Plane
    #except:
        #raise ImportError("FATAL: ENABLE_CALIBRATION is set, but Excalibur was not found!")

from .bbox import xywh2corners
from .enums import Singleton
from .geometry import vehicle_polygon_from_imu, point_rect_intersection
from .misc import copy_to_3d_floats


def estimate_ground_plane(world_points: np.ndarray, z_offset: float=0.0, force_up=False):
    gnss = copy_to_3d_floats(world_points)
    gnss[:, 2] -= z_offset
    svd = np.linalg.svd(gnss.T - np.mean(gnss, axis=0, keepdims=True).T)
    normal = svd[0][:, -1]
    if force_up and normal[2] > 0:
        normal, -normal
    return normal, np.mean(gnss, axis=0)


def estimate_ground_plane_by_cov(world_points):
        n = len(world_points)
        center = world_points.mean(axis=0)
        shifted = world_points - center
        cov = np.zeros((3, 3))
        cov[0, 0] = (shifted[:, 0] * shifted[:, 0]).sum()
        cov[0, 1] = (shifted[:, 0] * shifted[:, 1]).sum()
        cov[0, 2] = (shifted[:, 0] * shifted[:, 2]).sum()
        cov[1, 1] = (shifted[:, 1] * shifted[:, 1]).sum()
        cov[1, 2] = (shifted[:, 1] * shifted[:, 2]).sum()
        cov[2, 2] = (shifted[:, 2] * shifted[:, 2]).sum()
        cov /= n
        weighted_dir = np.zeros(3)
        detx = cov[1, 1] * cov[2, 2] - cov[1, 2] * cov[1, 2]
        dety = cov[0, 0] * cov[2, 2] - cov[0, 2] * cov[0, 2]
        detz = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[0, 1]
        detzz = cov[0, 2] * cov[1, 2] - cov[0, 1] * cov[2, 2]
        detyy = cov[0, 1] * cov[1, 2] - cov[0, 2] * cov[1, 1]
        detxx = cov[0, 1] * cov[0, 2] - cov[1, 2] * cov[0, 0]
        axis_dir = np.array([detx, detzz, detyy])
        weight = detx ** 2 if weighted_dir.dot(axis_dir) >= 0 else -(detx ** 2)
        weighted_dir += axis_dir * weight
        axis_dir = np.array([detzz, dety, detxx])
        weight = dety ** 2 if weighted_dir.dot(axis_dir) >= 0 else -(dety ** 2)
        weighted_dir += axis_dir * weight
        axis_dir = np.array([detyy, detxx, detz])
        weight = detz ** 2 if weighted_dir.dot(axis_dir) >= 0 else -(detz ** 2)
        weighted_dir += axis_dir * weight
        return weighted_dir / np.linalg.norm(weighted_dir), center


def gen_planes(k_mat: np.ndarray, box: np.ndarray):
    bbox = np.ones((4, 3))
    bbox[:, :2] = xywh2corners(box)
    k_inv = np.linalg.inv(k_mat)

    normals = []
    for i in range(4):
        p1 = bbox[i]
        p2 = bbox[(i + 1) % 4]
        p1 = k_inv @ p1
        p2 = k_inv @ p2
        normal = np.cross(p1, p2)
        normal /= np.linalg.norm(normal)
        normals.append(normal)

    return np.array(normals)[[3, 1, 2, 0]].T


def gen_lines(k_mat: np.ndarray, p: np.ndarray):
    k_inv = np.linalg.inv(k_mat)
    pi = np.ones(3)
    pi[:2] = p
    po = k_inv @ pi
    return po / np.linalg.norm(po)


def optimize_Q_plane(points_3d: np.ndarray, points_2d: np.ndarray, k_mat: np.ndarray):
    if points_3d.ndim == 3:
        points_3d_out = np.concatenate(points_3d, axis=0).T
    else:
        points_3d_out = np.copy(points_3d)
    planes_out = np.concatenate([gen_planes(k_mat, bbox) for bbox in points_2d], axis=1)
    optimizer = P2PL()
    optimizer.set_data(points_3d_out, planes_out)
    return optimizer.calibrate()


def optimize_Q_line(points_3d: np.ndarray, points_2d: np.ndarray, k_mat: np.ndarray):
    if points_3d.ndim == 3:
        points_3d_out = np.concatenate(points_3d, axis=0).T
    else:
        points_3d_out = np.copy(points_3d).T
    lines_out = [Line(point=np.zeros(3), direction=gen_lines(k_mat, p)) for p in points_2d]
    optimizer = P2L()
    optimizer.set_data(points_3d_out, lines_out)
    return optimizer.calibrate()


def optimize_Q_point(points_src: np.ndarray, points_dst: np.ndarray):
    optimizer = P2P()
    optimizer.set_points(points_src.T, points_dst.T)
    return optimizer.calibrate()


def estimate_3d_tf_svd(src, dst, weights=None):
    src_mean = src[:, :3].mean(axis=0)
    dst_mean = dst[:, :3].mean(axis=0)
    tf_src = src - src_mean
    tf_dst = dst - dst_mean
    if weights is not None:
        tf_src = tf_src * weights[:, None]
        tf_dst = tf_dst * weights[:, None]
    H = tf_src.T @ tf_dst
    U, S, VH = np.linalg.svd(H)
    R = VH.T @ U.T
    if np.linalg.det(R) < 0:
        VH[2, :] = VH[2, :] * -1.0
        R = VH.T @ U.T
    t = dst_mean - R @ src_mean
    res = np.zeros((3, 4))
    res[:3, :3] = R
    res[:3, 3] = t
    return res


def predict(data, times, dt, dry=False, dry_data=None, min_len=10):
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

    def fx(x, dt_):
        # state transition function - predict next state based on model
        x, y, v, a, yaw, yaw_rate = x.tolist()

        if abs(yaw_rate) < 0.0001:  # Driving straight
            return np.array([
                x + v * dt_ * cos(yaw),
                y + v * dt_ * sin(yaw),
                v + a * dt_,
                a,
                yaw + yaw_rate * dt_,
                yaw_rate
            ])
        return np.array([
            x + (v / yaw_rate) * (sin(yaw_rate * dt_ + yaw) - sin(yaw)),
            y + (v / yaw_rate) * (-cos(yaw_rate * dt_ + yaw) + cos(yaw)),
            v + a * dt_,
            a,
            (yaw + yaw_rate * dt_ + pi) % (2.0 * pi) - pi,
            yaw_rate
        ])


    def hx(x):
        # measurement function - convert state into a measurement
        # where measurements are [x_pos, y_pos]
        return np.array([x[0], x[1]])
    
    def limit_2pi(data_):
        data_ = np.fmod(data_, 2 * np.pi)
        data_[data_ > np.pi] -= 2 * np.pi
        data_[data_ < -np.pi] += 2 * np.pi
        return data_


    times_history = times[:min_len]
    # initial state
    lookahead = min(len(times_history) - 1, 5)
    diff0y = data[lookahead, 1] - data[0, 1]
    diff0x = data[lookahead, 0] - data[0, 0]
    theta0 = np.arctan2(diff0y, diff0x)
    yrt0 = 0.0
    v0 = np.sqrt(diff0y ** 2 + diff0x ** 2) / (times_history[lookahead] - times_history[0])
    a0 = 0.0

    # rotate and translate to (0,0) with orientation 0 to prevent issues with angles
    pos0 = data[0]
    c, s = np.cos(theta0), np.sin(theta0)
    rot = np.array([[c, -s], [s, c]])
    rot_I = np.linalg.inv(rot)
    data = (data - pos0).dot(rot)
    theta0_ = theta0
    theta0 = 0

    x = np.array([data[0, 0], data[0, 1], v0, a0, theta0, yrt0])

    # points = MyPoints(Q, (5,4), 6, alpha=.1, beta=2., kappa=-1)
    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    kf = UnscentedKalmanFilter(dim_x=6, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
    kf.x = x
    z_std = 0.50 ** 2
    kf.P = np.diag([z_std, z_std, 0.6, 10.0, 0.2, 0.5])
    kf.R = np.diag([z_std, z_std])

    def calculate_Q():
        gamma_jerk = np.array([
            1 / 6.0 * np.cos(kf.x[4]) * dt ** 3,
            1 / 6.0 * np.sin(kf.x[4]) * dt ** 3,
            1 / 2.0 * dt * dt,
            dt
        ])
        gamma_yaw_acc = np.array([
            1 / 2.0 * dt * dt,
            dt
        ])
        Q_jerk = np.outer(gamma_jerk, gamma_jerk) * 20.0 ** 2
        Q_yaw_acc = np.outer(gamma_yaw_acc, gamma_yaw_acc) * 2.0 ** 2
        Q = np.zeros((6, 6))
        Q[:4, :4] = Q_jerk
        Q[4:, 4:] = Q_yaw_acc
        return Q

    Qa = Q_discrete_white_noise(3, dt=dt, var=1.0)
    Qyrt = Q_discrete_white_noise(3, dt=dt, var=0.2)
    Qa_rows = [(0, 0), (1, 0), (2, 1), (3, 2)]
    Qyrt_rows = [(0, 0), (1, 0), (4, 1), (5, 2)]
    kf.Q = np.zeros((6, 6))
    for i in range(len(Qa_rows)):
        r1 = Qa_rows[i][0]
        r2 = Qa_rows[i][1]
        for j in range(len(Qa_rows)):
            c1 = Qa_rows[j][0]
            c2 = Qa_rows[j][1]
            kf.Q[r1, c1] += Qa[r2, c2]
    for i in range(len(Qyrt_rows)):
        r1 = Qyrt_rows[i][0]
        r2 = Qyrt_rows[i][1]
        for j in range(len(Qyrt_rows)):
            c1 = Qyrt_rows[j][0]
            c2 = Qyrt_rows[j][1]
            kf.Q[r1, c1] += Qyrt[r2, c2]
    np.set_printoptions(precision=3)
    # print(kf.Q)

    # actual loop over samples
    means = [kf.x]
    raw_data = [dry_data[0][:, :3]] if dry else []
    covs = [kf.P]
    times_out = [0.0]
    # m_id = 1
    num_steps = int(np.round((times[-1] - times[0]) / dt))
    cur_step = 1
    for filterstep in range(1, num_steps):
        kf.Q = calculate_Q()
        # np.set_printoptions(precision=3)
        # print(kf.Q)
        # print("-" * 50)
        # print()
        # print(kf.x)
        # print(kf.P)
        kf.predict()

        # if measurement available
        # if times[m_id] == filterstep:
        #     kf.update(data[m_id])
        #     m_id += 1
        # print(times[cur_step], times[0] + filterstep * dt)
        dry_chunk = Singleton.DUMMY
        if cur_step < len(times) and np.abs(times[cur_step] - (times[0] + filterstep * dt)) < dt:
            kf.update(data[cur_step])
            if dry:
                dry_chunk = dry_data[cur_step][:, :3]
            cur_step += 1

        means.append(kf.x)
        covs.append(kf.P)
        if dry:
            raw_data.append(dry_chunk)
        times_out.append(times[0] + filterstep * dt)

    means, covs, K = kf.rts_smoother(np.array(means), np.array(covs))

    res = np.array([np.concatenate([x, np.diagonal(P)]) for x, P in zip(means, covs)])

    # flip orientation if vehicle drives backwards
    if np.mean(res[:, 2]) < -0.5:
        # print("flipping orientation")
        res[:, 2] *= -1
        res[:, 3] *= -1
        res[:, 4] += np.pi

    # rotate and translate back to the original position
    res[:, :2] = res[:, :2].dot(rot_I) + pos0
    res[:, 4] = limit_2pi(res[:, 4] + theta0_)

    if dry:
        return res, np.array(times_out), raw_data
    else:
        return res, np.array(times_out)
    

def get_track_offset(vehicle_data, clusters, config, verbose=False):
        
    imu_offset = np.array([config.imu_x, config.imu_y, config.imu_z])

    gps = np.array([vehicle_polygon_from_imu(state,
                                            imu_offset[:2],
                                            config.vehicle_length, config.vehicle_width) for state in vehicle_data])

    def loss_fn(par):
        loss_acc = 0.0
        for gp, cluster in zip(gps, clusters):
            local_cluster = cluster[:, :2] + par[None, :]
            for p in local_cluster:
                if point_rect_intersection(p, gp):
                    continue
                else:
                    loss_acc += np.linalg.norm(gp - p[None, :], axis=1).min()
        return loss_acc

    res = minimize(loss_fn, np.array([0, 0]), method='Nelder-Mead', options={'fatol': 1e-4})
    if verbose:
        print("Offsets:", res.x, res.fun)
    return res.x


def estimate_box(cluster, prev_cluster, sensor_pos, dt, size_prior=None, use_simple_model=False):
    closest_point = cluster[np.argmin(np.linalg.norm(
        cluster[:, :2] - sensor_pos, axis=1))][:2]
    old_closest_point = prev_cluster[np.argmin(
        np.linalg.norm(prev_cluster[:, :2] - sensor_pos, axis=1))][:2]
    heading = closest_point - old_closest_point
    sensor_heading = closest_point - sensor_pos
    radial_vel_true = cluster[:, 3]
    radial_vecs = cluster[:, :2] - sensor_pos
    radial_vecs_norm = radial_vecs / \
                       np.linalg.norm(radial_vecs, axis=1, keepdims=True)
    radial_vel_est = np.dot(radial_vecs_norm, heading / dt)
    # print("radar heading match:", radial_vel_true / radial_vel_est)
    avg_radial = np.mean(radial_vecs_norm, axis=0)
    avg_radial = avg_radial / np.linalg.norm(avg_radial)
    heading_radial = np.dot(avg_radial, heading) * avg_radial
    heading_tangential = heading - heading_radial
    heading_radial_new = np.mean(radial_vel_true) * avg_radial
    heading = heading_tangential + heading_radial_new
    ref = "FRONT" if np.dot(heading, sensor_heading) < 0 else "BACK"
    ref_int = 5 if np.dot(heading, sensor_heading) < 0 else 1
    angle = np.arctan2(heading[1], heading[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ]).T
    points = np.copy(cluster[:, :2])
    cog = points.mean(axis=0)
    tf_points = (R @ (points - cog).T).T
    length = np.max(tf_points[:, 0]) - np.min(tf_points[:, 0])
    width = np.max(tf_points[:, 1]) - np.min(tf_points[:, 1])
    if length < 3.0:
        length = 3.0
    if width < 1.0:
        width = 1.0
    if size_prior is not None:
        length, width = size_prior
    # TODO: Optimize prior alignment
    if ref == "FRONT":
        ref_point = np.array([np.max(
            tf_points[0]), 0.5 * np.max(tf_points[:, 1]) + 0.5 * np.min(tf_points[:, 1])])
    else:
        ref_point = np.array([np.min(
            tf_points[0]), 0.5 * np.max(tf_points[:, 1]) + 0.5 * np.min(tf_points[:, 1])])
    itf_ref_point = (R.T @ ref_point) + cog
    state_dict = {"class": Classification.CAR_UNION, "prob": 0.5, "reference_point": ref,
                  "vals": {"YAW": (angle, 1.0), "X": (itf_ref_point[0], 1.0), "Y": (itf_ref_point[1], 1.0)},
                  "send_attrs": ["X", "Y", "YAW"], "send_attr_ints": [0, 1, 11],
                  "ref_int": ref_int, "cov_xy": 0.0}
    if not use_simple_model:
        state_dict["vals"]["LENGTH"] = (length, 1.0)
        state_dict["vals"]["WIDTH"] = (width, 1.0)
        state_dict["send_attrs"] += ["LENGTH", "WIDTH"]
        state_dict["send_attr_ints"] += [14, 15]
    return state_dict

def estimate_heading_from_cluster(shifted_cluster):
    y = shifted_cluster[:, 3]
    az = shifted_cluster[:, 4]
    X = np.array([np.cos(az), np.sin(az)]).T
    eta = np.array([p[5] for p in shifted_cluster]).mean() / np.array([p[6] for p in shifted_cluster]).mean()
    num_targets = len(shifted_cluster)
    ransac_count = min(num_targets * (num_targets - 1), 250)
    residuals = []
    for _ in range(ransac_count):
        idx = np.random.choice(num_targets, 2, replace=False)
        X_loc = X[idx]
        y_loc = y.reshape((-1, 1))[idx]
        p = np.linalg.inv(X_loc.T @ X_loc) @ X_loc.T @ y_loc
        residuals.append((np.abs(y - X @ p).mean(), idx, p))
    residual_min = np.array([r[0] for r in residuals]).min()
    best_params = residuals[np.array([r[0] for r in residuals]).argmin()][2]
    inliers = list(set(np.array([r[1] for r in residuals if r[0] < residual_min * 1.25]).flatten()))
    X = X[inliers]
    y = y[inliers]
    az = az[inliers]

    def single_estimate(x, vx, vy, vr, theta):
        return (vr - vx * np.cos(x) - vy * np.sin(x)) ** 2 + eta * (theta - x) ** 2

    def single_jacobian(x, vx, vy, vr, theta):
        outer = vr - vx * np.cos(x) - vy * np.sin(x)
        inner = vx * np.sin(x) - vy * np.cos(x)
        return 2 * outer * inner - 2 * eta * (theta - x)

    def compound_estimate(x, vx, vy, vr, theta):
        vel_term = (vr - vx * np.cos(x) - vy * np.sin(x)) ** 2
        angle_term = eta * (theta - x) ** 2
        return np.sum(vel_term + angle_term)

    def compound_jacobian(x, vx, vy, vr, theta):
        outer = vr - vx * np.cos(x) - vy * np.sin(x)
        fp = np.sum(2 * outer * (- np.cos(x)))
        sp = np.sum(2 * outer * (- np.sin(x)))
        return np.array([[fp, sp]])

    def optimize_single_step(x_in, alpha, beta):
        x0, vx, vy, vr, theta = x_in
        x1 = x0 - alpha / (single_jacobian(*x_in) ** 2 + beta) * single_jacobian(*x_in) * single_estimate(*x_in)
        return np.array([x1, vx, vy, vr, theta])

    def optimize_compound_step(x_in, alpha, beta):
        x, vx, vy, vr, theta = x_in
        res_0 = np.array([vx, vy])
        cj = compound_jacobian(*x_in)
        res_1 = res_0 - alpha / (cj.T @ cj + beta) * cj.flatten() * compound_estimate(*x_in)
        res_1 = res_1.flatten()
        return x, res_1[0], res_1[1], vr, theta

    a1 = 1e-2
    b1 = 1e-2
    a2 = 1e-2
    b2 = 1e-2

    iter0 = 0
    iter1 = 0
    iter0_max = 5
    d0 = None
    d0p = None
    diff = None
    while (diff is not None and diff < 1e-2) or iter0 < iter0_max:

        X_new = np.zeros(len(y))

        for i in range(len(y)):
            x_start = np.array([np.arctan2(best_params[1], best_params[0]), *best_params, y[i], az[i]])
            iter1 = 0
            iter1_max = 5
            d1 = None
            d1p = None
            diff = None
            while (diff is not None and diff < 1e-3) or iter1 < iter1_max:
                x_start = optimize_single_step(x_start, a1, b1)
                if d1 is None:
                    d1 = x_start[0]
                else:
                    d1p = d1
                    d1 = x_start[0]
                    diff = np.abs(d1 - d1p)
                iter1 += 1
            X_new[i] = x_start[0]

        x_start = (X_new, best_params[0], best_params[1], y, az)
        iter1 = 0
        iter1_max = 5
        d1 = None
        d1p = None
        diff = None
        while (diff is not None and diff < 1e-2) or iter1 < iter1_max:
            x_start = optimize_compound_step(x_start, a2, b2)
            if d1 is None:
                d1 = np.array([x_start[1], x_start[2]])
            else:
                d1p = d1
                d1 = np.array([x_start[1], x_start[2]])
                diff = np.linalg.norm(d1 - d1p)
            iter1 += 1
        best_params = np.array([x_start[1], x_start[2]])

        if d0 is None:
            d0 = best_params
        else:
            d0p = d1
            d1 = best_params
            diff = np.linalg.norm(d0 - d0p)
        iter0 += 1
    return np.arctan2(best_params[1], best_params[0])


def sort_cluster(cluster, ref):
    base_vecs = cluster[:, :3] - ref
    vertex_angles = np.arctan2(base_vecs.T[1], base_vecs.T[0])
    sorted_angles = np.sort(vertex_angles)
    if np.max(np.diff(sorted_angles)) >= np.pi:
        vertex_angles = np.array([alpha + 2 * np.pi if alpha < 0 else alpha for alpha in vertex_angles])
    sorted_vertex_indices = np.argsort(vertex_angles)
    return cluster[sorted_vertex_indices]


# From: Efficient L-shape Fitting of Laser Scanner Data for Vehicle Pose Estimation
def l_shaped_fit(cluster):
    nu = 1
    m = len(cluster)
    l_star = 1e10
    p = nu
    q = m - nu
    u = None
    m13 = cluster[:p, 0].sum()
    m14 = cluster[:p, 1].sum()
    m23 = cluster[p:, 1].sum()
    m24 = -cluster[p:, 0].sum()
    m33 = np.power(cluster[:p, 0], 2).sum() + np.power(cluster[p:, 1], 2).sum()
    m34 = (cluster[:p, 0] * cluster[:p, 1]).sum() - (cluster[p:, 0] * cluster[p:, 1]).sum()
    m44 = np.power(cluster[:p, 1], 2).sum() + np.power(cluster[p:, 0], 2).sum()
    M = np.array([
        [p, 0, m13, m14],
        [0, q, m23, m24],
        [m13, m23, m33, m34],
        [m14, m24, m34, m44],
    ])
    eigval, eigvec = np.linalg.eig(M[2:, 2:] - M[2:, :2].T @ (np.linalg.inv(M[:2, :2]) @ M[2:, :2]))
    # print("-" * 50)
    # print(eigval, eigvec)
    # TODO: Check if smallest eigenval or eigenval with smallest absolute value
    min_index = np.argmin(np.abs(eigval))
    eigval = eigval[min_index]
    eigvec = eigvec[min_index]
    if l_star > eigval:
        c = - np.linalg.inv(M[:2, :2]) @ (M[2:, :2] @ eigvec)
        l_star = eigval
        u = np.concatenate([c, eigvec], axis=0).flatten()
    while nu < m - 1:
        xn = cluster[nu, 0]
        yn = cluster[nu, 1]
        dm13 = xn
        dm14 = yn
        dm23 = -yn
        dm24 = xn
        dm33 = xn ** 2 - yn ** 2
        dm34 = 2 * xn * yn
        dm44 = yn ** 2 - xn **2
        dm = np.array([
            [1, 0, dm13, dm14],
            [0, -1, dm23, dm24],
            [dm13, dm23, dm33, dm34],
            [dm14, dm24, dm34, dm44],
        ])
        M = M + dm
        eigval, eigvec = np.linalg.eig(M[2:, 2:] - M[2:, :2].T @ (np.linalg.inv(M[:2, :2]) @ M[2:, :2]))
        # print(eigval, eigvec)
        min_index = np.argmin(np.abs(eigval))
        eigval = eigval[min_index]
        eigvec = eigvec[min_index]
        if l_star > eigval:
            c = - np.linalg.inv(M[:2, :2]) @ (M[2:, :2] @ eigvec)
            l_star = eigval
            u = np.concatenate([c, eigvec], axis=0).flatten()
        nu += 1
    return u


def estimate_bounding_box(cluster, u):
    xc = -u[2] * u[0] + u[3] * u[1]
    yc = -u[3] * u[0] - u[2] * u[1]
    theta = np.arctan2(u[3], u[2])
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    center = cluster[:, :2].mean(axis=0)
    cluster_copy = np.concatenate([cluster[:, :2], np.array([[xc, yc]])], axis=0)
    cluster_copy -= center
    cluster_copy = (R @ cluster_copy.T).T
    edge_point_1 = cluster[np.argmax(np.abs(cluster_copy[:-1, 0] - cluster_copy[-1, 0])), :2]
    edge_point_2 = cluster[np.argmax(np.abs(cluster_copy[:-1, 1] - cluster_copy[-1, 1])), :2]
    return [edge_point_1, np.array([xc, yc]), edge_point_2, edge_point_2 + (edge_point_1 - np.array([xc, yc]))]


def fit_box(cluster, p, draw_commands=False):
    sc = sort_cluster(np.copy(cluster), p)
    u = l_shaped_fit(sc)
    box = estimate_bounding_box(sc, u)
    if draw_commands:
        draw_commands = []
        for i in range(3):
            a = box[i % 3]
            b = box[(i + 1) % 3]
            draw_commands.append(([a[0], b[0]], [a[1], b[1]], (1.0, 1.0, 1.0, 1.0)))
        return draw_commands
    else:
        return box


def fit_aligned_bounding_box(cluster, draw_commands=False):
    sc = cv2.convexHull(np.copy(cluster[:, :2]).astype("float32"))
    rect = cv2.minAreaRect(sc)
    box = cv2.boxPoints(rect).astype(float)
    if draw_commands:
        draw_commands = []
        for i in range(4):
            a = box[i % 4]
            b = box[(i + 1) % 4]
            draw_commands.append(([a[0], b[0]], [a[1], b[1]], (1.0, 1.0, 1.0, 1.0)))
        return draw_commands
    else:
        return box
