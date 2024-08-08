from pathlib import Path

import numpy as np
import yaml
from numba import njit

from .utils import geometry as gl


@njit(fastmath=True)
def image_uv2world_fast_multiple(uv, K_inv, R_inv, t_inv, n_cc, b_cc):
    uv_ = (K_inv @ uv.T).T
    ray_len = np.dot(b_cc, n_cc) / np.dot(uv_, n_cc)
    p_ = np.expand_dims(ray_len, axis=-1) * uv_
    ret = (R_inv @ p_.T).T + t_inv
    return ret


@njit(fastmath=True)
def image_uv2world_fast_single(uv, K_inv, R_inv, t_inv, n_cc, b_cc):
    uv_ = K_inv @ uv
    ray_len = np.dot(b_cc, n_cc) / np.dot(uv_, n_cc)
    p_ = ray_len * uv_
    ret = R_inv @ p_ + t_inv
    return ret


@njit(fastmath=True)
def world_xy2image_fast_single(xyz, K, R, t, n, b):
    xyz[2] = np.dot(b[:2] - xyz[:2], n[:2]) / n[2] + b[2]
    uvz_ = R @ xyz + t
    uvz_ = uvz_ / uvz_[2]
    ret = (K @ uvz_)[:2]
    return ret


@njit(fastmath=True)
def world_xyz2image_fast_single(xyz, K, R, t):
    uvz_ = R @ xyz + t
    uvz_ = uvz_ / uvz_[2]
    ret = (K @ uvz_)[:2]
    return ret


class CameraCalibration:

    @classmethod
    def load(cls, filename, calibration_version_override=None):
        calibration_path = Path(filename)
        if not calibration_path.exists():
            raise ValueError(f"Failed to load calibration. Reason: File \"{str(calibration_path)}\" does not exist")
        with open(calibration_path, "r") as f:
            camera_metadata = yaml.safe_load(f)
            calibration_metadata = camera_metadata["calibration"]
            calibration_version = calibration_metadata[
                "current_version"] if calibration_version_override is None else calibration_version_override
            calibration = calibration_metadata["data"][calibration_version]
            if "width" not in calibration_metadata:
                calibration["width"] = float(camera_metadata["camera_parameter"]["genicam_features"]["Width"])
                calibration["height"] = float(camera_metadata["camera_parameter"]["genicam_features"]["Height"])
            else:
                calibration["width"] = calibration_metadata["width"]
                calibration["height"] = calibration_metadata["height"]
            if "rotate_cw" in camera_metadata["mono_processing"].keys() or "rotate_ccw" in camera_metadata[
                "mono_processing"].keys():
                calibration["width"], calibration["height"] = calibration["height"], calibration["width"]
            return cls(calibration)

    @classmethod
    def load_local(cls, K, R, t, n, b, w, h):
        param_dict = {
            "K": K,
            "R": R,
            "t": t,
            "n": n,
            "b": b,
            "width": w,
            "height": h
        }
        return cls(param_dict)

    def __init__(self, param_dict):
        self.K = np.array(param_dict["K"])
        self.R = np.array(param_dict["R"])
        self.t = np.array(param_dict["t"])
        self.n = np.array(param_dict["n"])
        self.b = np.array(param_dict["b"])

        if self.n[2] < 0:
            self.n = -self.n

        # Ground plane in camera coordinate system
        self.n_cc = self.R @ self.n
        self.b_cc = (self.R @ self.b) + self.t

        self.K_inv = np.linalg.inv(self.K)
        self.R_inv = self.R.T
        # Inverse translation equals the camera position
        self.t_inv = - self.R_inv @ self.t

        bottom_center = np.array([param_dict["width"] / 2.0, param_dict["height"]])
        uv_dir = np.array([0.0, -1.0])
        self.camera_heading_vec = self.dir_uv2dir_xy(uv_dir, bottom_center)
        self.camera_heading_angle = gl.vec2angle(self.camera_heading_vec)

        self.width = param_dict["width"]
        self.height = param_dict["height"]

    def world2image_uv(self, xyz):
        assert xyz.shape[-1] == 3
        uvz_ = (self.R @ xyz.T).T + self.t
        uvz_ = uvz_ / uvz_[:, 2:]
        ret = (self.K @ uvz_.T)[:2].T
        if ret.shape[0] == 1:
            ret = ret[0]
        return ret

    def world2camera(self, xyz):
        assert xyz.shape[-1] == 3
        ret = (self.R @ xyz.T).T + self.t
        if ret.shape[0] == 1:
            ret = ret[0]
        return ret

    def world_xyz2image_uv(self, xyz):
        return self.world2image_uv(xyz)

    def world_xyz2image_uv_single(self, xyz):
        return world_xyz2image_fast_single(xyz, self.K, self.R, self.t)

    def world_xy2image_uv(self, xy):
        assert xy.shape[-1] == 2
        xyz = gl.to_homogenous_batch(xy)
        xyz[:, 2] = np.dot(self.b[:2] - xyz[:, :2], self.n[:2]) / self.n[2] + self.b[2]
        return self.world2image_uv(xyz)

    def world_xy2camera(self, xy):
        assert xy.shape[-1] == 2
        xyz = gl.to_homogenous_batch(xy)
        xyz[:, 2] = np.dot(self.b[:2] - xyz[:, :2], self.n[:2]) / self.n[2] + self.b[2]
        return self.world2camera(xyz)

    def image_uv2world(self, uv):
        return image_uv2world_fast_multiple(uv, self.K_inv, self.R_inv, self.t_inv, self.n_cc, self.b_cc)

    def image_uv2world_xyz(self, uv):
        assert uv.shape[-1] == 2
        ret = self.image_uv2world(gl.to_homogenous_batch(uv))
        if ret.shape[0] == 1:
            ret = ret[0]
        assert ret.shape[-1] == 3
        return ret

    def image_uv2world_xy(self, uv):
        assert uv.shape[-1] == 2
        ret = self.image_uv2world(gl.to_homogenous_batch(uv))
        if ret.shape[0] == 1:
            ret = ret[0]
        assert ret.shape[-1] == 3
        return ret[..., :2]

    def image_uv2world_xy_single(self, uv):
        buf = np.array([uv[0], uv[1], 1.0])
        return image_uv2world_fast_single(buf, self.K_inv, self.R_inv, self.t_inv, self.n_cc, self.b_cc)[:2]

    def world_xy2image_uv_single(self, xy):
        buf = np.array([xy[0], xy[1], 1.0])
        return world_xy2image_fast_single(buf, self.K, self.R, self.t, self.n, self.b)[:2]

    def image_uv2world_xy_multiple(self, uv):
        buf = np.array([uv[:, 0], uv[:, 1], [1.0] * len(uv)]).T
        return image_uv2world_fast_multiple(buf, self.K_inv, self.R_inv, self.t_inv, self.n_cc, self.b_cc)[:, :2]

    def extend_height_at_xy(self, xy, z_offset=0.0):
        assert xy.shape[-1] == 2
        uvz = gl.to_homogenous_batch(xy)
        # If n[2] < 0: self.b - z_offset * self.n / np.linalg.norm(self.n)
        b_temp = self.b + z_offset * self.n / np.linalg.norm(self.n)
        uvz[:, 2] = np.dot(b_temp[:2] - uvz[:, :2], self.n[:2]) / self.n[2] + b_temp[2]
        return uvz

    def dir_uv2dir_xy(self, dir_uv, at):
        world_at = self.image_uv2world_xy(at)
        world_at_duv = self.image_uv2world_xy(at + gl.normalize(dir_uv))
        return gl.normalize(world_at_duv - world_at)

    def dir_uv2dir_xy_single(self, dir_uv, at):
        world_at = self.image_uv2world_xy_single(at)
        world_at_duv = self.image_uv2world_xy_single(at + gl.normalize(dir_uv))
        return gl.normalize(world_at_duv - world_at)

    def dir_xy2dir_uv(self, dir_xy, at, tf_at=True):
        image_at = self.world_xy2image_uv(at) if tf_at else np.copy(at)
        if not tf_at:
            at = self.image_uv2world_xy_single(at)
        image_at_duv = self.world_xy2image_uv(at + gl.normalize(dir_xy) * 0.1)
        return gl.normalize(image_at_duv - image_at)

    def dir_xy2dir_uv_single(self, dir_xy, at, tf_at=True):
        image_at = self.world_xy2image_uv_single(at) if tf_at else np.copy(at)
        if not tf_at:
            at = self.image_uv2world_xy_single(at)
        image_at_duv = self.world_xy2image_uv_single(at + gl.normalize(dir_xy) * 0.1)
        return gl.normalize(image_at_duv - image_at)

    def generate_base_coeffs(self):
        # Ground plane equation: a * x + b * y + c * z + d = 0
        #
        #          ((r11 - r13 * a / c) * x + (r12 - r13 * b / c) * y - r13 * d / c + t1)
        # u = fx * ---------------------------------------------------------------------- + cx
        #          ((r31 - r33 * a / c) * x + (r32 - r33 * b / c) * y - r33 * d / c + t3)
        #
        # converts to:
        #          alpha * x + beta * y + gamma
        # u = fx * ----------------------------- + cx
        #          delta * x + epsilon * y + eta

        a, b, c = self.n
        x0, y0, z0 = self.b
        d = - (a * x0 + b * y0 + c * z0)
        alpha_x = self.R[0, 0] - self.R[0, 2] * a / c
        beta_x = self.R[0, 1] - self.R[0, 2] * b / c
        gamma_x = - self.R[0, 2] * d / c + self.t[0]
        alpha_y = self.R[1, 0] - self.R[1, 2] * a / c
        beta_y = self.R[1, 1] - self.R[1, 2] * b / c
        gamma_y = - self.R[1, 2] * d / c + self.t[1]
        delta = self.R[2, 0] - self.R[2, 2] * a / c
        epsilon = self.R[2, 1] - self.R[2, 2] * b / c
        eta = - self.R[2, 2] * d / c + self.t[2]
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        return [alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y, delta, epsilon, eta, fx, fy, cx, cy]
