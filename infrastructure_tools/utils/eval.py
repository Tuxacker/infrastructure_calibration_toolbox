import cv2
import numpy as np

from ..calibration_provider import CameraCalibration
from ..trackers import TargetTracker
from .bbox import xywh2corners
from .geometry import line_intersection2d, vehicle_polygon_from_imu, point_rect_intersection, dist_to_gp, sort_polygon
from .optim import get_track_offset
from .time import find_matches


def measure_calibration_offsets(world_track: np.ndarray, image_track: np.ndarray, k_mat: np.ndarray, calibration_mat: np.ndarray, imu_offset: np.ndarray, n:float, b: float, w: int, h: int, vehicle_length: float, vehicle_width: float):

    local_calib = CameraCalibration.load_local(k_mat, calibration_mat[:3, :3], calibration_mat[:3, 3], n, b, w, h)

    offsets = []
    camera_distances = []

    camera_position = local_calib.t_inv

    for world_point, box in zip(world_track, image_track):
        ground_points = vehicle_polygon_from_imu(world_point, imu_offset, vehicle_length, vehicle_width)
        box_corners = xywh2corners(box)
        bottom_left = local_calib.image_uv2world_xy_single(box_corners[0])
        bottom_right = local_calib.image_uv2world_xy_single(box_corners[1])
        edge_dir_box = bottom_left - bottom_right
        edge_dir_box /= np.linalg.norm(edge_dir_box)
        edge_dir_perp = np.array([-edge_dir_box[1], edge_dir_box[0]])
        polygon_dists = []
        for p in ground_points:
            polygon_dists.append(
                np.linalg.norm(
                    line_intersection2d(
                        bottom_left,
                        bottom_left + edge_dir_box,
                        p,
                        p + edge_dir_perp
                        )
                     - p)
                )
        offsets.append(np.min(polygon_dists))
        closest_point = np.zeros(3)
        closest_point[:2] = ground_points[np.argmin(np.array(polygon_dists))]
        closest_point[2] = np.dot(b[:2] - closest_point[:2], n[:2]) / n[2] + b[2]
        camera_distances.append(offsets[-1] / np.linalg.norm(closest_point - camera_position) * 100)

    offsets = np.array(offsets)
    camera_distances = np.array(camera_distances)

    return offsets, camera_distances


def evaluate_radar(mat, radar_data, imu_data, config, sensor_name, dt=1/15, dbscan_eps=4.0, min_length=10, full=False, silent=False, verbose=False):
    if verbose:
        import matplotlib.pyplot as plt

    imu_offset = np.array([config.imu_x, config.imu_y, config.imu_z])

    rmat = mat[:3, :3]
    tvec = mat[:3, 3]
    radar_pos = np.copy(tvec).flatten()

    def tf(points):
        return (rmat @ points.T).T + tvec.reshape((1, 3))


    tracker = TargetTracker(store_tracks=True, use_3d=True)
    tracker.kwargs["dbscan_eps"] = dbscan_eps
    # tracker.kwargs["assoc_thr"] = 5.0

    all_targets = []
    for t in sorted(radar_data.keys()):
        tracker.unlock_for_cur_step()
        filtered_data = np.array(radar_data[t])
        filtered_data = filtered_data[~np.isnan(filtered_data[:, -1])][:, :-1]
        filtered_data[:, :3] = tf(filtered_data[:, :3])
        tracker.add_targets(filtered_data, t)
        all_targets.append(filtered_data)

    tracks = tracker.get_sanitized_tracks()

    offsets = []
    offset_weights = []

    full_dists = []
    full_ref_dists = []
    full_ratios = []
    outlier_dists = []

    for track_id, track in enumerate(tracks):
        timestamps, target_chunks = track
        if len(timestamps) < min_length:
            continue
        dets = {t: tc for t, tc in zip(timestamps, target_chunks)}
        matched_pairs = find_matches(imu_data, dets, dt)
        p1 = np.array([a[1][:2] for a in matched_pairs])
        p1_full = [a[1] for a in matched_pairs]
        p2 = np.array([a[0][:, :2].mean(axis=0) for a in matched_pairs])
        p2_full = [a[0][:, :2] for a in matched_pairs]
        if p1.ndim != 2 or p2.ndim != 2:
            print(p1.shape, p2.shape)
            continue
        dists = np.linalg.norm(p2 - p1, axis=1)
        if dists.mean() < 5.0:
            if silent:
                est_offset = get_track_offset(p1_full, p2_full, config, verbose=verbose)
                offsets.append(est_offset)
                offset_weights.append(len(p1_full))
            new_dists = []
            new_ref_dists = []
            inlier_ratio = []
            for adma_state, cluster in zip(p1_full, p2_full):
                adma_box = vehicle_polygon_from_imu(adma_state,
                                                    imu_offset[:2],
                                                    config.vehicle_length,
                                                    config.vehicle_width)
                closest_adma_point = adma_box[np.linalg.norm(adma_box - radar_pos[None, :2], axis=1).argmin()]
                closest_cluster_point = cluster[np.linalg.norm(cluster - radar_pos[None, :2], axis=1).argmin()]
                if not 15 < np.linalg.norm(closest_adma_point - radar_pos[:2]) < 50:
                    continue
                new_dists.append(np.linalg.norm(closest_adma_point - closest_cluster_point))
                # FIXME: this should be done in 3D
                # FIXME: Use radial distance as reference?
                new_ref_dists.append(np.linalg.norm(closest_adma_point - radar_pos[:2]))
                num_targets = len(cluster)
                num_inliers = 0
                for p in cluster:
                    if point_rect_intersection(p, adma_box[:4, :2]):
                        num_inliers += 1
                    else:
                        outlier_dists.append(
                            (dist_to_gp(p, adma_box[:4, :2]), np.linalg.norm(closest_adma_point - radar_pos[:2])))
                inlier_ratio.append(num_inliers / num_targets)
                full_ratios.append((num_inliers, num_targets))
            if len(new_dists) == 0:
                if verbose:
                    print("All dists not in range")
                continue
            new_dists = np.array(new_dists)
            rel_errors = np.abs(new_dists / np.array(new_ref_dists)) * 100
            inlier_ratio = np.array(inlier_ratio)
            full_dists.append(new_dists)
            full_ref_dists.append(new_ref_dists)
            if not silent:
                if verbose:
                    print("-" * 25)
                    print(new_dists.mean(), new_dists.max())
                    print(rel_errors.mean(), rel_errors.max())
                    print(inlier_ratio.mean(), inlier_ratio.max())
                    if inlier_ratio.max() > 0.75:
                        seq_len = len(timestamps)
                        plt.axis("equal")
                        for i in range(0, seq_len, 10):
                            adma_state, cluster = p1_full[i], p2_full[i]
                            adma_box = vehicle_polygon_from_imu(adma_state,
                                                    imu_offset[:2],
                                                    config.vehicle_length,
                                                    config.vehicle_width)
                            adma_box = np.concatenate([adma_box, adma_box[None, 0]], axis=0)
                            color = plt.cm.inferno(i / seq_len)
                            plt.plot(adma_box.T[0], adma_box.T[1], color=color)
                            plt.scatter(cluster.T[0], cluster.T[1], color=color)
                            if len(cluster) > 2:
                                local_cluster = sort_polygon(cluster[:, :2])
                                hull = cv2.convexHull(np.copy(local_cluster).astype(np.single)).reshape((-1, 2))
                                plt.fill(hull.T[0], hull.T[1], color=color)
                        plt.savefig(f"radar_track_{track_id}.pdf", dpi=300)
                        plt.close()

    if silent:
        offsets = np.array(offsets).reshape((-1, 2))
        offset_weights = np.array(offset_weights).flatten()
        weights = np.array(offset_weights)
        weights = weights / weights.sum()
        correction_term = (np.array(offsets) * weights[:, None]).sum(axis=0)
        if verbose:
            print("Correction:", correction_term)
        return correction_term

    if full:
        full_dists = np.concatenate(full_dists)
        full_ref_dists = np.concatenate(full_ref_dists)
        inlier_count = np.array([a[0] for a in full_ratios]).sum()
        total_count = np.array([a[1] for a in full_ratios]).sum()
        outlier_errors = np.array([a[0] for a in outlier_dists])
        outlier_dists = np.array([a[1] for a in outlier_dists])
        rel_errors = full_dists / full_ref_dists * 100
        ratio = inlier_count / total_count
        if verbose:
            print(f"Evaluation result for {sensor_name}:")
            print("Distance error (m,w):", full_dists.mean(), full_dists.max())
            print("Relative error (m,w):", rel_errors.mean(), rel_errors.max())
            print("Inlier ratio:", ratio)
            print("Outlier average error:", outlier_errors.mean())
            print("-" * 10)
            print("Best solution:")
            print(mat)
            sorted_indices = np.argsort(full_ref_dists)
            xvals = full_ref_dists[sorted_indices]
            yvals_abs = full_dists[sorted_indices]
            yvals_rel = rel_errors[sorted_indices]
            plt.plot(xvals, yvals_abs)
            plt.savefig(f"radar_{sensor_name}_abs.pdf", dpi=300)
            plt.close()
            plt.plot(xvals, yvals_rel)
            plt.savefig(f"radar_{sensor_name}_rel.pdf", dpi=300)
            plt.close()
            sorted_indices = np.argsort(outlier_dists)
            plt.plot(outlier_dists[sorted_indices], outlier_errors[sorted_indices])
            plt.savefig(f"radar_{sensor_name}_outliers.pdf", dpi=300)
            plt.close()
