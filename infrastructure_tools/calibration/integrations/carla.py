from pathlib import Path
from yaml import safe_load

import numpy as np

def load_camera_parameters(config_dict: dict, **kwargs):
    K = np.eye(3)
    focal = config_dict["camera_width"] / (2.0 * np.tan(config_dict["camera_fov"] * np.pi / 360.0))
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = config_dict["camera_width"] / 2
    K[1, 2] = config_dict["camera_height"] / 2

    c_y = np.cos(np.radians(config_dict["camera_yaw"]))
    s_y = np.sin(np.radians(config_dict["camera_yaw"]))
    c_r = 1
    s_r = 0
    c_p = np.cos(np.radians(config_dict["camera_pitch"]))
    s_p = np.sin(np.radians(config_dict["camera_pitch"]))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = config_dict["camera_x"]
    matrix[1, 3] = config_dict["camera_y"]
    matrix[2, 3] = config_dict["camera_z"]
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = c_y * s_p * c_r + s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = s_y * s_p * c_r - c_y * s_r
    matrix[2, 0] = -s_p
    matrix[2, 1] = c_p * s_r
    matrix[2, 2] = c_p * c_r

    matrix = np.linalg.inv(matrix)

    # Conversion from left-handed into right-handed coordinate system
    camera_pos = -matrix[:3, :3].T @ matrix[:3, 3]
    gt_approx = np.array([-matrix[1], matrix[2], matrix[0], matrix[3]]).reshape((4, 4))
    gt_approx[1:3, 2] = -gt_approx[1:3, 2]
    gt_approx[:3, 3] = (-gt_approx[:3, :3] @ camera_pos.reshape((3, 1))).flatten()

    return K, gt_approx