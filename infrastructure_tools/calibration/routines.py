from typing import List, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from ..calibration_provider import CameraCalibration
from ..utils.bbox import xywh2corners
from ..utils.eval import measure_calibration_offsets
from ..utils.geometry import line_intersection2d, vehicle_volume_from_imu
from ..utils.math import reproject_3d_to_2d, get_camera_position, rodrigues_to_mat, transform_world_to_camera, find_loops
from ..utils.misc import copy_to_2d_floats, copy_to_3d_floats
from ..utils.optim import estimate_ground_plane, optimize_Q_line
from .definitions import CalibrationConfig, CalibrationData, Correspondence, Hypothesis


def merge_hypotheses(calibration_data: CalibrationData, hypothesis_list: List[Hypothesis], filter_inliers: bool, verbose: bool=False):
    timestamp_set = set()
    t = []
    world_points = []
    box_points = []
    timestamp_overlap = False
    opt = []
    if verbose:
        print(f"Trying to merge {len(hypothesis_list)} hypotheses")
    for hypothesis in hypothesis_list:
        track_pair: Correspondence = calibration_data[hypothesis.data_id]
        inlier_ids = hypothesis.inlier_ids
        track_timestamps = list(track_pair.t[inlier_ids]) if filter_inliers else list(track_pair.t)
        if verbose and len(set(track_timestamps) & timestamp_set) > 0:
            print("Found duplicate timestamps!")
            timestamp_overlap = True
        timestamp_set = timestamp_set | set(track_timestamps)
        t += track_timestamps
        world_points.append(track_pair.gnss[inlier_ids] if filter_inliers else track_pair.gnss)
        box_points.append(track_pair.box[inlier_ids] if filter_inliers else track_pair.box)
        if track_pair.opt is not None:
            opt.append(track_pair.opt[inlier_ids] if filter_inliers else track_pair.opt)
    world_points = np.concatenate(world_points, axis=0)
    box_points = np.concatenate(box_points, axis=0)
    if len(opt) > 0:
        opt = np.concatenate(opt, axis=0)
    else:
        opt = None
    return Correspondence(np.array(t), world_points, box_points, opt), timestamp_overlap
    

def get_overlap_score(points: np.ndarray, heatmap: np.ndarray, r_vec: np.ndarray, t_vec: np.ndarray, config: CalibrationConfig):
    projected_points = reproject_3d_to_2d(points, r_vec, t_vec, config.k_matrix, config.distortion_coefficients)
    reshaped_projection = np.repeat(projected_points[:, None, :], heatmap.shape[0], axis=1)
    reshaped_boxes = np.repeat(heatmap[None, :, :], reshaped_projection.shape[0], axis=0)
    distance_field = np.linalg.norm(reshaped_boxes - reshaped_projection, axis=-1)
    return ((distance_field < config.maximal_distance_in_heatmap).sum(axis=1) > 0).sum() / distance_field.shape[0]


def process_points_for_optim_single(track_pair: Correspondence, config: CalibrationConfig, r_vec: np.ndarray, t: np.ndarray,
                                    n: np.ndarray, b_in: np.ndarray, inlier_ids: Union[List, np.ndarray]=None,
                                    use_top_point: bool=False, verbose: bool=False):
    if inlier_ids is None:
        inlier_ids = np.arange(track_pair.gnss.shape[0], dtype=int)

    initial_camera_position = get_camera_position(r_vec, t)
    b = np.copy(b_in)

    if use_top_point:
        if verbose:
            print("Using top point")
        b += np.array([0, 0, config.vehicle_height])

    if use_top_point:
        def tf(x):
            min_i = np.argmax(np.linalg.norm(x - initial_camera_position[np.newaxis, :], axis=1))
            return x[min_i]
    else:
        def tf(x):
            min_i = np.argmin(np.linalg.norm(x - initial_camera_position[np.newaxis, :], axis=1))
            return x[min_i]

    corner_indices = [1, 2, 5, 6] if use_top_point else [0, 3, 4, 7]
    imu_offset = np.array([config.imu_x, config.imu_y, config.imu_z])
    input_points_3d = np.array([tf(vehicle_volume_from_imu(track_pair.gnss[_id],
                                                       imu_offset,
                                                       config.vehicle_length,
                                                       config.vehicle_width,
                                                       config.vehicle_height)[corner_indices]) for _id in inlier_ids])

    points_2d_filtered = np.array([box for box in track_pair.box[inlier_ids]])
    timestamp_inliers = track_pair.t[inlier_ids]

    local_calib = CameraCalibration.load_local(config.k_matrix, rodrigues_to_mat(r_vec), t.flatten(), n, b, config.camera_width,
                                               config.camera_height)

    def tf_and_filter(p):
        corners = xywh2corners(p)
        box_world = []
        for corner in corners:
            world_position = local_calib.image_uv2world_xyz(corner)
            cc = local_calib.R @ world_position + local_calib.t
            if cc[2] < 0 or np.linalg.norm(cc) > 100:
                return None
            box_world.append(world_position[:2])
        return np.array(box_world)

    validity_mask = np.array([True if tf_and_filter(box) is not None else False for box in points_2d_filtered])
    timestamps_filtered = timestamp_inliers[validity_mask]
    points_3d_from_2d = np.array([tf_and_filter(box) for box in points_2d_filtered if tf_and_filter(box) is not None])
    points_2d_matching = np.array([box for box in points_2d_filtered if tf_and_filter(box) is not None])
    input_points_3d = input_points_3d[validity_mask]
    input_points_len = len(input_points_3d)
    assoc_2d_points = []
    assoc_3d_points = []
    assoc_timestamps = []

    if verbose:
        print("Points before filter:", input_points_len)

    for p_vehicle, box_transformed, box_original, _, _t in zip(input_points_3d, points_3d_from_2d, points_2d_matching,
                                               range(input_points_len), timestamps_filtered):
        point = p_vehicle[:2]

        box = box_transformed[:, :2]

        base_direction = box[1] - box[0] if not use_top_point else box[2] - box[3]
        left_corner = line_intersection2d(point, point + base_direction, box[0], box[3])
        right_corner = line_intersection2d(point, point + base_direction, box[1], box[2])
        ratio = np.linalg.norm(point - left_corner) / np.linalg.norm(right_corner - left_corner)
        mapped_point = box[0] + ratio * (box[1] - box[0]) if not use_top_point else box[3] + ratio * (
                box[2] - box[3])
        
        # TODO: Clarify lifting threshold and parametrize maximum distance to estiamted camera position

        # if np.linalg.norm(point - mapped_point) / np.linalg.norm(
        #         point - initial_camera_position[:2]) < config.lifting_threshold:
        if np.linalg.norm(point - mapped_point) < config.vehicle_length:
            if np.linalg.norm(point - initial_camera_position[:2]) > 60:
                continue
            assoc_3d_points.append(p_vehicle)
            assoc_timestamps.append(_t)
            if use_top_point:
                assoc_2d_points.append(
                    np.array([box_original[0] + (ratio - 0.5) * box_original[2], box_original[1] - box_original[3] / 2]))
            else:
                assoc_2d_points.append(
                    np.array([box_original[0] + (ratio - 0.5) * box_original[2], box_original[1] + box_original[3] / 2]))
 
    if verbose:
        print("Points after filter:", len(assoc_3d_points))

    return np.array(assoc_3d_points), np.array(assoc_2d_points), np.array(assoc_timestamps)


def is_correspondence_valid(correspondence: Correspondence, config: CalibrationConfig):
        gnss = correspondence.gnss
        box = correspondence.box

        # Filter out tracks that don't have enough data points
        if gnss.shape[0] < config.minimal_sequence_length:
            # print(f"Not enough data points! ({imu_data_points.shape[0]})")
            return False
        # Filter out short tracks in pixel coordinates
        elif np.linalg.norm(box[:, :2].max(axis=0) - box[:, :2].min(axis=0)) < config.minimal_image_track_length:
            # print(
            #     f"Not enough uv distance! ({np.linalg.norm(box_data_points[:, :2].max(axis=0) - box_data_points[:, :2].min(axis=0)):.2f})")
            return False
        # Filter out short tracks in world coordinates
        elif np.linalg.norm(gnss[:, :2].max(axis=0) - gnss[:, :2].min(axis=0)) < config.minimal_world_track_length:
            # print(
            #     f"Not enough world distance! ({np.linalg.norm(imu_data_points[:, :2].max(axis=0) - imu_data_points[:, :2].min(axis=0)):.2f})")
            return False
        else:
            return True
        

def estimate_transform(gnss_data: np.ndarray, box_data: np.ndarray, config: CalibrationConfig):
        if config.use_ransac:
            ret, r_vec, t_vec, inliers = cv2.solvePnPRansac(gnss_data, box_data, config.k_matrix,
                                                            config.distortion_coefficients,
                                                            flags=cv2.SOLVEPNP_EPNP,
                                                            reprojectionError=config.maximal_projection_error,
                                                            iterationsCount=config.maximal_ransac_iterations)
        else:
            ret, r_vec, t_vec, _ = cv2.solvePnPGeneric(gnss_data, box_data, config.k_matrix,
                                                       config.distortion_coefficients, flags=cv2.SOLVEPNP_EPNP)
            inliers = np.arange(len(gnss_data)).astype(int)
        if not ret:
            return None, None, None
        else:
            if not config.use_ransac:
                r_vec, t_vec = r_vec[0], t_vec[0]
            return r_vec, t_vec, inliers.flatten()
        

def estimate_hypothesis(calibration_data: CalibrationData, index: int, config: CalibrationConfig, verbose: bool=False):
        corr = calibration_data[index]
        t = corr.t
        gnss_data = copy_to_3d_floats(corr.gnss)
        box_data = copy_to_2d_floats(corr.box)

        if not is_correspondence_valid(calibration_data[index], config):
            # print(f"ID {pair_index} is not a valid index")
            return None, None, None

        if config.use_yolov5_labels:
            box_data[:, 0] *= config.camera_width
            box_data[:, 1] *= config.camera_height

        r_vec, t_vec, inliers = estimate_transform(gnss_data, box_data, config)

        if r_vec is None or t_vec is None:
            if verbose:
                print(f"ID {index} did not yield a solution")
            return None, None, None

        camera_position = get_camera_position(r_vec, t_vec)
        distance_to_track = np.linalg.norm(
            calibration_data.gnss_data[:, :2] - camera_position.reshape((1, -1))[:, :2], axis=1)

        # Filter out camera positions too far away from driven path
        # noinspection PyArgumentList
        if verbose:
            print(f"ID {index} minimal distance between vehicle and infrastructure data:", distance_to_track.min())
        if distance_to_track.min() > config.maximal_camera_distance_to_driven_path:
            if verbose:
                print(f"ID {index} is not next to a driven path")
            return None, None, None

        # Filter out camera positions below the road plane
        if config.filter_negative_heights and camera_position[2] < gnss_data[:, 2].min():
            if verbose:
                print(f"ID {index} is below ground")
            return None, None, None

        gnss_reprojected = reproject_3d_to_2d(gnss_data, r_vec, t_vec, config.k_matrix, config.distortion_coefficients)
        projection_errors = np.linalg.norm(gnss_reprojected - box_data, axis=1)

        # Filter out transforms with excessive median projection error
        if np.sort(projection_errors)[projection_errors.shape[0] // 2] > config.maximal_median_projection_error:
            if verbose:
                print(f"ID {index} exceeds the maximal allowed median reprojection error")
            return None, None, None

        if verbose:
            print(f" ID {index} is estimated at p_camera = {camera_position}")

        return r_vec, t_vec, inliers


def estimate_multiple_hypotheses(calibration_data: CalibrationData, config: CalibrationConfig, verbose: bool=False):
    hypotheses = []
    hypothesis_count = 0
    for i in range(calibration_data.pair_count):
        r_vec, t_vec, inliers = estimate_hypothesis(calibration_data, i, config, verbose)
        if r_vec is not None:
            if verbose:
                print(f"Added hypothesis with ID {hypothesis_count:06d} ({len(inliers)} inliers)")
            hypotheses.append(
                Hypothesis(_id=hypothesis_count, data_id=i, r_vec=r_vec, t=t_vec, inlier_ids=inliers))
            hypothesis_count += 1
    return hypotheses


def calculate_scores_and_build_laplacian(calibration_data: CalibrationData, hypotheses: List[Hypothesis], config: CalibrationConfig, verbose: bool=False):
    hypothesis_count = len(hypotheses)
    laplacian = np.zeros((hypothesis_count, hypothesis_count), dtype=int)
    hypothesis_scores = {}

    for i in range(hypothesis_count):
        for j in range(i, hypothesis_count):
            if i == j:
                continue
            hypothesis_i = hypotheses[i]
            hypothesis_j = hypotheses[j]

            # TODO: Parametrize toggle of similarity metrics

            # imu_data_points_i_in_camera_j = transform_world_to_camera(
            #     calibration_data[hypothesis_i.data_id].gnss, hypothesis_j.R, hypothesis_j.t)
            # imu_data_points_j_in_camera_i = transform_world_to_camera(
            #     calibration_data[hypothesis_j.data_id].gnss, hypothesis_i.R, hypothesis_i.t)
            # transform_ij_inliers = (imu_data_points_i_in_camera_j[:, 2] > 0) & (
            #         np.linalg.norm(imu_data_points_i_in_camera_j, axis=1) < config.maximal_sensor_range)
            # transform_ji_inliers = (imu_data_points_j_in_camera_i[:, 2] > 0) & (
            #         np.linalg.norm(imu_data_points_j_in_camera_i, axis=1) < config.maximal_sensor_range)
            # outlier_score_ij = 1 - transform_ij_inliers.sum() / transform_ij_inliers.shape[0]
            # outlier_score_ji = 1 - transform_ji_inliers.sum() / transform_ji_inliers.shape[0]
            # overlap_score_ji = get_overlap_score(calibration_data[hypothesis_i.data_id].gnss,
            #                                     calibration_data.get_heatmap(), hypothesis_i.R,
            #                                     hypothesis_i.t, config)
            # overlap_score_ij = get_overlap_score(calibration_data[hypothesis_j.data_id].gnss,
            #                                     calibration_data.get_heatmap(), hypothesis_j.R,
            #                                     hypothesis_j.t, config)



            # if outlier_score_ij < config.outlier_score_threshold \
            #     and outlier_score_ji < config.outlier_score_threshold \
            #     and overlap_score_ij > config.overlap_score_threshold \
            #     and overlap_score_ji > config.overlap_score_threshold:
            #     pass

            r_mat_i = Rotation.from_matrix(rodrigues_to_mat(hypothesis_i.r_vec))
            r_mat_j = Rotation.from_matrix(rodrigues_to_mat(hypothesis_j.r_vec)).inv()
            concatenated_rotation = r_mat_j * r_mat_i
            rotational_similarity = np.trace(concatenated_rotation.as_matrix()) / 3
            if rotational_similarity > config.rotational_similarity_threshold:
                if verbose:
                    # print(f"Mapping {i:03d}->{j:03d}: ov_j = {overlap_score_ij:.4f}, ov_i = {overlap_score_ji:.4f}, out_j = {outlier_score_ij:.4f}, out_i = {outlier_score_ji:.4f}, rsim = {rotational_similarity:.4f}")
                    print(f"Mapping {i:03d}->{j:03d}: rsim = {rotational_similarity:.4f}")
                laplacian[i, i] += 1
                laplacian[j, j] += 1
                laplacian[i, j] = -1
                laplacian[j, i] = -1
                if i in hypothesis_scores.keys():
                    hypothesis_scores[i].append(rotational_similarity)
                else:
                    hypothesis_scores[i] = [rotational_similarity]
                if j in hypothesis_scores.keys():
                    hypothesis_scores[j].append(rotational_similarity)
                else:
                    hypothesis_scores[j] = [rotational_similarity]

    return hypothesis_scores, laplacian


def initial_scoring_and_clustering(calibration_data: CalibrationData, hypotheses: List[Hypothesis], config: CalibrationConfig, disable_clustering: bool=False, use_single_hypothesis: bool=False, verbose: bool=False):
    hypothesis_scores, laplacian = calculate_scores_and_build_laplacian(calibration_data, hypotheses, config, verbose)

    loops = find_loops(laplacian)

    if verbose:
        print(f"Found loops: {loops}")

    filtered_hypotheses = []

    if verbose:
        import matplotlib.pyplot as plt

    for loop in loops:
        if len(loops) == 1 and len(loop) == 1:
            filtered_hypotheses.append(loop)
        elif len(loop) == 1 and use_single_hypothesis:
            filtered_hypotheses.append(loop)
        elif len(loop) < 2:
            continue
        elif disable_clustering:
            filtered_hypotheses.append(loop)
        else:
            camera_positions = np.array(
                [get_camera_position(hypotheses[index].r_vec, hypotheses[index].t) for index in loop])
            clustering = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples).fit(
                camera_positions)
            num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            for label in range(num_clusters):
                filtered_hypotheses.append(np.array(loop, dtype=int)[clustering.labels_ == label].tolist())
            if verbose:
                for label in set(clustering.labels_):
                    if label == -1:
                        plt.scatter(*camera_positions[clustering.labels_ == label].T[:2], color="gray")
                    else:
                        plt.scatter(*camera_positions[clustering.labels_ == label].T[:2])
    
    if verbose:
        plt.axis("equal")
        plt.grid()
        plt.xlabel("Position X [m]")
        plt.ylabel("Position Y [m]")
        plt.savefig("hypothesis_clusters.pdf", dpi=300)

    if verbose:
        print(f"Filtered hypotheses: {filtered_hypotheses}")

    return filtered_hypotheses, hypothesis_scores


def hypothesis_refinement(hypothesis: Hypothesis, scores: dict[int, List[float]], correspondence: Correspondence, config: CalibrationConfig, from_merged: bool, verbose: bool=False):
    _id = hypothesis._id
    h_score = np.array(scores[_id]).mean() if not from_merged and _id in scores else -1.0

    if verbose:
        print(f"ID: {_id}, Camera position: {get_camera_position(hypothesis.r_vec, hypothesis.t)}")
        print(f"Score mean: {h_score}")

    n_local, b_local = estimate_ground_plane(correspondence.gnss, config.imu_z)
    local_rvec, local_tvec = np.copy(hypothesis.r_vec), np.copy(hypothesis.t)
    result = None
    used_timestamps = None

    # TODO: Parametrize number of iterations

    for _ in range(1):
        points_3d, points_2d, used_timestamps = process_points_for_optim_single(correspondence, config,
                                                                                local_rvec, local_tvec,
                                                                                n_local, b_local,
                                                                                hypothesis.inlier_ids, verbose=verbose)
        if verbose:
            print(f"Number of optimizable/total points/inlier length: {points_3d.shape[0]}/{len(correspondence.box)}/{len(hypothesis.inlier_ids)}")

        if points_3d.shape[0] > 2:
            result = optimize_Q_line(points_3d, points_2d, config.k_matrix)

            # Sanity Check for debugging

            # rvec_new, tvec_new = cv2.solvePnPRefineLM(points_3d, points_2d, config.k_matrix, None,
            #                                                       np.copy(hypothesis.r_vec), np.copy(hypothesis.t))
            # print(f"Alternative position: {get_camera_position(rvec_new, tvec_new)}")

            # TODO: Check if alternative solution is feasible and choose the better result

            # rvec_new, tvec_new = cv2.solvePnPRefineLM(points_3d, points_2d, config.k_matrix, None,
            #                                             np.copy(hypothesis.r_vec), np.copy(hypothesis.t))
            # print(f"Alternative position: {get_camera_position(rvec_new, tvec_new)}")

            if result is not None and result.success:
                optim_solution = np.linalg.inv(result.calib.getMatrix())
                local_tvec = np.copy(optim_solution[:3, 3]).reshape((3, 1))
                local_rvec = cv2.Rodrigues(optim_solution[:3, :3])[0].reshape((3, 1))
            else:
                if verbose:
                    print("Optimization failed")
                    print(f"Result: {result}")
                    print("Attempting OpenCV optimization")
                local_rvec_new, local_tvec_new = cv2.solvePnPRefineLM(points_3d, points_2d, config.k_matrix, None,
                                                            np.copy(local_rvec), np.copy(local_tvec))
                distance = np.linalg.norm(get_camera_position(local_rvec_new, local_tvec_new) - get_camera_position(local_rvec, local_tvec))
                if verbose:
                    print(f"Alternative position: {get_camera_position(local_rvec_new, local_tvec_new)}")
                    print(f"Distance to previous solution: {distance:.2f} m")
                # TODO: Add parameter to config
                if distance > 10.0:
                    if verbose:
                        print("Discarding OpenCV result")
                    return None, None, None, None
                optim_solution = np.eye(4)
                optim_solution[:3, :3] = cv2.Rodrigues(local_rvec_new)[0].reshape((3,3))
                optim_solution[:3, 3] = local_tvec_new.flatten()
                local_rvec, local_tvec = local_rvec_new, local_tvec_new

        else:
            if verbose:
                print("Not enough refinement points")
            return None, None, None, None

    old_position = get_camera_position(hypothesis.r_vec, hypothesis.t).flatten()
    new_position = -optim_solution[:3, :3].T @ optim_solution[:3, 3]

    # TODO: Parametrize maximum allowed result shift for each iteration
    ignore_shift = True

    if ignore_shift or np.linalg.norm(new_position - old_position) < 10:
        imu_offset = np.array([config.imu_x, config.imu_y])
        offsets_and_dists = measure_calibration_offsets(correspondence.gnss, correspondence.box, config.k_matrix, optim_solution, imu_offset, n_local, b_local, config.camera_width, config.camera_height, config.vehicle_length, config.vehicle_width)

        if verbose:
            print(f"Camera position new: {- optim_solution[:3, :3].T @  optim_solution[:3, 3]}")

        return (optim_solution, n_local, b_local), offsets_and_dists, used_timestamps, h_score
    else:
        if verbose:
            print(f"New camera position deviates too much! {old_position}<=>{new_position}")
        return None, None, None, None
        
