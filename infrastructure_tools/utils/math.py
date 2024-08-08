from typing import List, Union
import cv2
import numpy as np

from .misc import copy_to_3d_floats


def rodrigues_to_mat(r_vec: np.ndarray):
    return cv2.Rodrigues(r_vec)[0]


def rodrigues_t_to_mat(r_vec: np.ndarray, t_vec: np.ndarray):
    mat = np.eye(4)
    mat[:3, :3] = rodrigues_to_mat(r_vec)
    mat[:3, 3] = t_vec.flatten()
    return mat


def reproject_3d_to_2d(points: np.ndarray, r_vec: np.ndarray, t_vec: np.ndarray, k_mat: np.ndarray, dist_coeffs: Union[List, np.ndarray]):
    reprojected_world_data_points = cv2.projectPoints(np.copy(copy_to_3d_floats(points)), r_vec, t_vec, k_mat, dist_coeffs)[0]
    return reprojected_world_data_points.reshape((-1, 2))


def transform_world_to_camera(points, r_vec, t_vec):
    return (rodrigues_to_mat(r_vec) @ copy_to_3d_floats(points).T).T + t_vec.reshape((1, 3))


def get_camera_position(r_vec: np.ndarray, t_vec: np.ndarray):
    return (- rodrigues_to_mat(r_vec).T @ t_vec).flatten()


def get_camera_position_4x4(mat: np.ndarray):
    return (- mat[:3, :3].T @ mat[:3, 3]).flatten()


def find_loops(laplacian: np.ndarray):
    adj = -laplacian
    adj[adj < 0] = 0
    visited_mask = np.zeros(adj.shape[0], dtype=int)
    loops = []
    _id = 1
    while (visited_mask == 0).sum() > 0:
        index = np.argmax(visited_mask == 0)
        visited_mask[index] = _id
        to_visit = adj[index]
        to_visit[index] = 1
        counter = 0
        while True:
            visited = np.copy(to_visit)
            current_indices = np.nonzero(to_visit)[0]
            for i in current_indices:
                visited |= adj[i]
            if counter >= adj.shape[0]:
                visited_mask[visited != 0] = - _id
                break
            elif (to_visit - visited).sum() == 0:
                visited_mask[visited != 0] = _id
                loops.append(np.nonzero(visited)[0].tolist())
                break
            else:
                to_visit = visited
                counter += 1
        _id += 1
    return loops
