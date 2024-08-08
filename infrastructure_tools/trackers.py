from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Dict, Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, NumbaPerformanceWarning
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from .utils.enums import Classification
from .utils.geometry import normalize_angle
from .utils.optim import fit_aligned_bounding_box


@njit(fastmath=True)
def bbox_diou(box1, box2):
    eps = 1e-7

    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1[:4], box2[:4]
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

    inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0) * \
            max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    cw = max(b1_x2, b2_x2) - min(b1_x1, b2_x1)
    ch = max(b1_y2, b2_y2) - min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    return iou - rho2 / c2, iou


@njit(fastmath=True)
def get_dir(boxes, s, sc):
    boxes[:, 1] += boxes[:, 3] / 2
    diffs = boxes[1:, :2] - boxes[:-1, :2]
    if len(diffs) > 1:
        used_hist = min(len(diffs), s)
        return sc[-used_hist:] @ diffs[-used_hist:]
    else:
        return diffs[0]


T = 0.1

F = np.array([
    [1, T],
    [0, 1]
])

H = np.array([
    [1, 0]
])


def filter_step_ctr(z, x, P, Q, R):
    # print("xi", x)
    xp = F @ x.reshape((2, 1))
    # print("xp", xp)
    Pp = F @ P @ F.T + Q
    if z is None:  # or np.abs(normalize_angle(z - H @ xp)) > np.pi / 8:
        return xp, Pp
    K = Pp @ H.T * 1 / (H @ Pp @ H.T + R)
    xn = xp + K * (z - H @ xp)
    Pn = (np.eye(2) - np.outer(K, H)) @ Pp
    xn[0] = normalize_angle(xn[0])
    clip_limit = np.pi / 2
    xn[1] = np.clip(xn[1], -clip_limit, clip_limit)
    # print("xn", xn)
    return xn.flatten(), Pn


@njit(fastmath=True)
def filter_step_cy(z, x, P, Q, R):
    xp = x
    Pp = P + Q
    if z is None:  # or np.abs(normalize_angle(z - H @ xp)) > np.pi / 8:
        return xp, Pp
    K = Pp * 1 / (Pp + R)
    residual = normalize_angle(z - xp)
    residual = max(residual, -np.pi / 2 * T)
    residual = min(residual, np.pi / 2 * T)
    xn = xp + K * residual
    Pn = (1 - K) * Pp
    xn = normalize_angle(xn)
    return xn, Pn


@njit(fastmath=True)
def filter_step_cy_adaptive(z, x, P, Q, R, alpha):
    xp = x
    Pp = P + Q
    Rn = R
    if z is None:  # or np.abs(normalize_angle(z - H @ xp)) > np.pi / 8:
        return xp, Pp, Rn
    residual = normalize_angle(z - xp)
    residual = max(residual, -np.pi / 2 * T)
    residual = min(residual, np.pi / 2 * T)
    Rn = alpha * Rn + (1 - alpha) * residual ** 2
    K = Pp * 1 / (Pp + Rn)
    xn = xp + K * residual
    Pn = (1 - K) * Pp
    xn = normalize_angle(xn)
    return xn, Pn, Rn


class MeasType(Enum):
    INIT = 1
    REG = 2
    STATIC = 3
    COPY = 4


class HeadingType(Enum):
    UNAVAILABLE = 1
    EMPTY = 2
    PREV_CUR = 3
    PREV_NEXT = 4
    SMOOTH = 5


class MetadataType(Enum):
    UNAVAILABLE = 1
    REFERENCE = 2


@dataclass
class MeasMetadata:
    reference_point: str


@dataclass
class TrackMetadata:
    start_point: np.ndarray
    total_displacement: np.ndarray = field(default_factory=lambda: np.zeros(2))
    length: int = 1
    yaw_state: float = 0  # np.ndarray = np.zeros(2)
    yaw_p: float = 0  # np.ndarray = np.eye(2)
    yaw_r: float = 0
    yaw_init: bool = False
    yaw_updated: bool = False


@dataclass
class Measurement:
    box: np.ndarray
    classification: Classification
    timestamp: int
    flag: MeasType
    heading: np.ndarray
    heading_type: HeadingType
    metadata: MeasMetadata = field(default_factory=lambda: MeasMetadata(""))
    metadata_type: MetadataType = MetadataType.UNAVAILABLE
    world_front: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class Estimate:
    full_box: np.ndarray
    dir_vector: np.ndarray
    object_id: int
    classification: Classification
    displacement: np.ndarray
    dir_vector_filtered: np.ndarray
    mask: Optional[np.ndarray] = None


def get_boxes(meas_arr: List[Measurement], last_n=None):
    return [item.box for item in meas_arr] if last_n is None else [item.box for item in meas_arr[-last_n:]]


def get_points(meas_arr: List[Measurement]):
    return np.array([item.world_front for item in meas_arr])


def get_flags(meas_arr: List[Measurement], func=None, last_n=None):
    if func is None:
        return [item.flag for item in meas_arr] if last_n is None else [item.flag for item in meas_arr[-last_n:]]
    else:
        return [func(item.flag) for item in meas_arr] if last_n is None else [func(item.flag) for item in
                                                                              meas_arr[-last_n:]]


def get_headings(meas_arr: List[Measurement]):
    return [item.heading for item in meas_arr]


def get_heading_types(meas_arr: List[Measurement]):
    return [item.heading_type for item in meas_arr]


def get_timestamps(meas_arr: List[Measurement]):
    return [item.timestamp for item in meas_arr]


def zero_hdg():
    return np.zeros(2)


class BoxTracker:

    def __init__(self, queue_size=10, smoothing=3, smoothing_coeff=0.9, store_tracks=False, logger=None,
                 class_config=None, to_world=None, to_world_dir=None):
        self.storage: Dict[int, List[Measurement]] = {}
        self.track_metadata: Dict[int, TrackMetadata] = {}
        self.qs = queue_size
        self.s = smoothing
        self.sc = np.flip(np.cumprod(smoothing_coeff * np.ones(self.s)))
        self.added_meas_in_cur_step = False
        self.logger = logger
        self.static_threshold = 1.0
        self.decay_after = 3
        self.current_mapped_ids = None
        self.store_tracks = store_tracks
        self.max_assoc_thr_dict = {class_: cfg["max_assoc_thr"] for class_, cfg in class_config.items() if
                                   class_config is not None and "max_assoc_thr" in cfg.keys()}
        self.use_copy_scaling_factor = True
        self.static_threshold = 0.1
        self.iou_func = bbox_diou

        def f(xa, xb):
            x1 = np.copy(xa)
            x2 = np.copy(xb)
            x1[1] += x1[3] / 2
            x2[1] += x2[3] / 2
            return np.linalg.norm(x1[:2] - x2[:2])

        self.default_displacement_func = f
        self.displacement_functions = {class_: cfg["displacement_function"] for class_, cfg in class_config.items() if
                                       class_config is not None and "displacement_function" in cfg.keys()}

        def df_proxy(xa, xb, class_in):
            if class_in in self.displacement_functions.keys():
                return self.displacement_functions[class_in](xa, xb)
            else:
                return self.default_displacement_func(xa, xb)

        def to_world_dummy(unused):
            return np.zeros(2)

        def to_world_dir_dummy(unused1, unused2):
            return np.zeros(2)

        self.displacement_function = df_proxy
        self.to_world = to_world
        if self.to_world is None:
            self.to_world = to_world_dummy
        self.to_world_dir = to_world_dir
        if self.to_world_dir is None:
            self.to_world_dir = to_world_dir_dummy
        self.track_storage = []
        self.cur_index = 0
        self.ids_mapped_from_input = {}

        # self.sigma_v = np.pi / 3 / 10
        #
        # self.P = np.eye(2) * np.pi / 12
        #
        # self.Q = np.array([
        #     [0.25 * T ** 4, 0.5 * T ** 3],
        #     [0.5 * T ** 3, T ** 2]
        # ]) * self.sigma_v ** 2
        #
        # self.R = np.pi / 10

        self.sigma_y = np.pi / 3

        self.P = np.pi / 12 * 4

        self.default_var = 1.0

        self.Q = T ** 2 * self.sigma_y ** 2

        self.R = np.pi / 10

        self.alpha = 0.5

    def incr_index(self):
        self.cur_index = (self.cur_index + 1) % 1024
        while self.cur_index in self.storage.keys():
            self.cur_index = (self.cur_index + 1) % 1024

    def unlock_for_cur_step(self):
        self.added_meas_in_cur_step = False
        self.current_mapped_ids = None
        self.ids_mapped_from_input = None
        for meta in self.track_metadata.values():
            meta.yaw_updated = False

    def extrapolate_track(self, i):
        if i in self.storage.keys():
            n = len(self.storage[i])
            if n == 1:
                return self.storage[i][-1].box
            elif n == 2:
                return self.storage[i][-1].box + (self.storage[i][-1].box - self.storage[i][-2].box)
            else:
                d1 = self.storage[i][-1].box - self.storage[i][-2].box
                d2 = self.storage[i][-2].box - self.storage[i][-3].box
                return self.storage[i][-1].box + d1 + (d1 - d2)

    def add_measurements(self, boxes: Union[List, np.ndarray], timestamp: int):
        if self.added_meas_in_cur_step:
            if self.logger is not None:
                self.logger.warn("Another measurement was already added during this time step, skipping!")
            return

        if type(boxes) == list:
            boxes = np.array(boxes)

        mapped_ids = []
        ids_mapped_from_input = {}

        cur_size, input_size = len(self.storage), boxes.shape[0]

        # No history, no new inputs
        if input_size == 0 and cur_size == 0:
            pass
        # No inputs, but history present => extrapolation
        elif input_size == 0 and cur_size > 0:
            for key in self.storage.keys():
                measurement = Measurement(self.extrapolate_track(key), self.storage[key][-1].classification, timestamp,
                                          MeasType.COPY, zero_hdg(), HeadingType.UNAVAILABLE)
                measurement.world_front = self.to_world(
                    np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                self.storage[key].append(measurement)
                self.track_metadata[key].total_displacement += (measurement.world_front - self.storage[key][
                    -2].world_front)
                self.track_metadata[key].length += 1
        # No history => track setup
        elif input_size > 0 and cur_size == 0:
            for input_idx, box in enumerate(boxes):
                measurement = Measurement(box, Classification(int(box[4])), timestamp, MeasType.INIT, zero_hdg(),
                                          HeadingType.UNAVAILABLE)
                measurement.world_front = self.to_world(
                    np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                self.storage[self.cur_index] = [measurement]
                self.track_metadata[self.cur_index] = TrackMetadata(measurement.world_front)
                mapped_ids.append(self.cur_index)
                ids_mapped_from_input[self.cur_index] = input_idx
                self.incr_index()
        # Generic case => associate new measurements
        else:
            cost_mat = np.zeros((cur_size, input_size))
            box_dict = self.get_last_boxes()
            keys = sorted(box_dict.keys())
            box_arr = [box_dict[k] for k in keys]

            remaining_row_indices = [keys[i] for i in range(cur_size)]
            remaining_col_indices = list(range(input_size))

            for row in range(cur_size):
                for col in range(input_size):
                    # cost_mat[row, col] = np.linalg.norm(boxes[col] - box_arr[row])
                    diou, iou = self.iou_func(boxes[col], box_arr[row])
                    if iou < 1e-6:
                        cost_mat[row, col] = 10
                    else:
                        cost_mat[row, col] = 1 - diou
            # Row: indices of objects of previous frame, col: indices of objects of current frame
            row_ids, col_ids = linear_sum_assignment(cost_mat)
            for row, col in zip(row_ids, col_ids):
                # Consider the edge case, where one or multiple objects appear and the same amount disappears
                # TODO: Check if old track was copied, increase limit accordingly
                # cost = cost_mat[row, col]
                # current_size = np.prod(self.box_history[keys[row]][-1][2:4])
                # input_size = np.prod(boxes[col, 2:4])
                # ratio = np.sqrt(input_size / current_size)
                # ratio = 1 / ratio if ratio < 1 else ratio
                # TODO: Relative scaling
                # print(f"map {row}->{col}: {np.linalg.norm(boxes[col] - self.box_history[keys[row]][-1])}")

                # if self.max_assoc_thr is not None:
                #     dist_cost = np.linalg.norm(boxes[col][:2] - box_arr[row][:2])
                #     copy_mask = np.flip(np.array([f == MeasType.COPY for f in self.flags[keys[row]]]))
                #     copy_scaling_factor = 1
                #     if self.use_copy_scaling_factor:
                #         if np.all(copy_mask):
                #             copy_scaling_factor = len(copy_mask)
                #         elif not np.any(copy_scaling_factor):
                #             copy_scaling_factor = 1
                #         else:
                #             copy_scaling_factor = copy_mask.astype(int).argmin() + 1
                #     if type(self.max_assoc_thr) == float and self.max_assoc_thr * copy_scaling_factor < dist_cost:
                #         continue
                #     elif self.max_assoc_thr(boxes[col][1]) * copy_scaling_factor < dist_cost:
                #         continue
                if cost_mat[row, col] > 1:
                    continue
                cur_class = Classification(int(boxes[col][4]))
                displacement = self.displacement_function(boxes[col], self.storage[keys[row]][-1].box, cur_class)
                if cur_class in self.max_assoc_thr_dict.keys() is not None and displacement > self.max_assoc_thr_dict[
                    cur_class]:
                    continue
                flag = MeasType.STATIC if displacement < self.static_threshold else MeasType.REG
                if flag == MeasType.REG:
                    current_heading, heading_flag = self.get_single_dir_vector(keys[row])
                    measurement = Measurement(boxes[col], cur_class, timestamp, flag, current_heading, heading_flag)
                    measurement.world_front = self.to_world(
                        np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                    self.storage[keys[row]].append(measurement)
                    # print("before1:", self.track_metadata[keys[row]].total_displacement)
                    self.track_metadata[keys[row]].total_displacement = self.track_metadata[
                                                                            keys[row]].total_displacement + (
                                                                                self.storage[keys[row]][
                                                                                    -1].world_front -
                                                                                self.storage[keys[row]][
                                                                                    -2].world_front)
                    self.track_metadata[keys[row]].length += 1
                    # print("after1:", self.track_metadata[keys[row]].total_displacement)
                else:
                    measurement = Measurement(boxes[col], cur_class, timestamp, flag, zero_hdg(),
                                              HeadingType.UNAVAILABLE)
                    measurement.world_front = self.to_world(
                        np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                    self.storage[keys[row]].append(measurement)
                    # print("before2:", self.track_metadata[keys[row]].total_displacement)
                    self.track_metadata[keys[row]].total_displacement = self.track_metadata[
                                                                            keys[row]].total_displacement + (
                                                                                self.storage[keys[row]][
                                                                                    -1].world_front -
                                                                                self.storage[keys[row]][
                                                                                    -2].world_front)
                    self.track_metadata[keys[row]].length += 1
                    # print("after2:", self.track_metadata[keys[row]].total_displacement)
                mapped_ids.append(keys[row])
                ids_mapped_from_input[keys[row]] = col
                remaining_row_indices.remove(keys[row])
                remaining_col_indices.remove(col)

            if len(remaining_row_indices) > 0:
                for key in remaining_row_indices:
                    measurement = Measurement(self.extrapolate_track(key), self.storage[key][-1].classification,
                                              timestamp, MeasType.COPY, zero_hdg(), HeadingType.UNAVAILABLE)
                    measurement.world_front = self.to_world(
                        np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                    self.storage[key].append(measurement)
                    self.track_metadata[key].total_displacement += (measurement.world_front - self.storage[key][
                        -2].world_front)
                    self.track_metadata[key].length += 1

            if len(remaining_col_indices) > 0:
                for index in remaining_col_indices:
                    measurement = Measurement(boxes[index], Classification(int(boxes[index][4])), timestamp,
                                              MeasType.INIT, zero_hdg(), HeadingType.UNAVAILABLE)
                    measurement.world_front = self.to_world(
                        np.array([measurement.box[0], measurement.box[1] + measurement.box[3] / 2]))
                    self.storage[self.cur_index] = [measurement]
                    self.track_metadata[self.cur_index] = TrackMetadata(measurement.world_front)
                    mapped_ids.append(self.cur_index)
                    ids_mapped_from_input[self.cur_index] = col
                    self.incr_index()

        self.remove_unlikely_tracks()

        self.trim_history()

        self.current_mapped_ids = mapped_ids
        self.ids_mapped_from_input = ids_mapped_from_input

        self.added_meas_in_cur_step = True

        # Cleans up tracks which have not enough history yet
        _ = self.get_dir_vectors(cleanup=True)
        # print("Flags after add:", [f"{i}: {get_flags(track)}" for i, track in self.storage.items()])
        # print("Hdgs after add:", [f"{i}: {get_heading_types(track)}" for i, track in self.storage.items()])

    def get_current_input_mappings(self):
        return self.ids_mapped_from_input
    
    def get_current_mapped_ids(self):
        return self.current_mapped_ids

    def get_points_sorted(self):
        return [get_points(self.storage[_id]) for _id in self.current_mapped_ids]

    def get_total_displacement_sorted(self):
        return [(np.linalg.norm(self.track_metadata[_id].total_displacement), self.track_metadata[_id].length) for _id
                in self.current_mapped_ids]

    def get_single_dir_vector(self, key, no_smoothing=False):
        local_qs = 10 if self.qs is None else None
        mask = get_flags(self.storage[key], lambda x: x == MeasType.INIT or x == MeasType.REG, local_qs)
        boxes = np.array(get_boxes(self.storage[key], local_qs))[mask]
        if len(boxes) >= 2:
            if no_smoothing:
                return get_dir(boxes[-2:], self.s, self.sc), HeadingType.PREV_CUR
            new_heading = get_dir(boxes, self.s, self.sc)
            if len(boxes) > 2:
                heading_type = HeadingType.SMOOTH
                if self.storage[key][-1].heading_type != HeadingType.SMOOTH:
                    self.storage[key][-1].heading = new_heading
                    self.storage[key][-1].heading_type = heading_type
                return new_heading, heading_type
            else:
                assert new_heading.ndim == 1 and len(new_heading) == 2
                heading_type = HeadingType.PREV_CUR
                if self.storage[key][-1].heading_type not in [HeadingType.SMOOTH, HeadingType.PREV_CUR]:
                    self.storage[key][-1].heading = new_heading
                    self.storage[key][-1].heading_type = heading_type
                return new_heading, heading_type
        else:
            return zero_hdg(), HeadingType.UNAVAILABLE

    def filter_dir(self, key):
        if self.track_metadata[key].yaw_init:
            if not self.track_metadata[key].yaw_updated:
                last_dir, type_ = self.get_single_dir_vector(key, no_smoothing=True)
                if type_ == HeadingType.UNAVAILABLE or np.linalg.norm(last_dir) < 1e-6:
                    z = None
                else:
                    at = self.storage[key][-1].box
                    at = np.array([at[0], at[1] + at[3] / 2])
                    last_dir = self.to_world_dir(last_dir, at)
                    z = np.arctan2(last_dir[1], last_dir[0])
                # print("key", key)
                # print(self.track_metadata[key].yaw_state / np.pi * 180)
                self.track_metadata[key].yaw_state, self.track_metadata[key].yaw_p, self.track_metadata[
                    key].yaw_r = filter_step_cy_adaptive(z,
                                                         self.track_metadata[
                                                             key].yaw_state,
                                                         self.track_metadata[
                                                             key].yaw_p,
                                                         self.Q,
                                                         self.R,
                                                         self.alpha)
                self.track_metadata[key].yaw_updated = True
                # print(self.track_metadata[key].yaw_state / np.pi * 180)
                # print(self.track_metadata[key].yaw_p)
        else:
            last_dir, type_ = self.get_single_dir_vector(key, no_smoothing=True)
            if type_ == HeadingType.UNAVAILABLE or np.linalg.norm(last_dir) < 1e-6:
                return
            else:
                at = self.storage[key][-1].box
                at = np.array([at[0], at[1] + at[3] / 2])
                last_dir = self.to_world_dir(last_dir, at)
                z = np.arctan2(last_dir[1], last_dir[0])
                self.track_metadata[key].yaw_state = z  # np.array([z, 0.0])
                self.track_metadata[key].yaw_p = self.P
                self.track_metadata[key].yaw_r = self.R
                self.track_metadata[key].yaw_init = True
                # print("key", key)
                # print(self.track_metadata[key].yaw_state / np.pi * 180)
                # print(self.track_metadata[key].yaw_p)

    def get_filter_dir(self, key, return_var=False):
        if self.track_metadata[key].yaw_init and self.track_metadata[key].yaw_updated:
            yaw = self.track_metadata[key].yaw_state  # [0]
            return np.array([np.cos(yaw), np.sin(yaw)]) if not return_var else np.array([np.cos(yaw), np.sin(yaw)]), \
                self.track_metadata[key].yaw_p
        else:
            return self.get_single_dir_vector(key)[0] if not return_var else self.get_single_dir_vector(key)[
                0], self.default_var

    def force_update_filtered_dirs(self):
        for _id in self.track_metadata.keys():
            self.filter_dir(_id)

    def get_filtered_dirs_sorted(self, return_var=False):
        for key in self.current_mapped_ids:
            self.filter_dir(key)
        return [self.get_filter_dir(_id, return_var) for _id in self.current_mapped_ids]

    def get_dir_vectors(self, cleanup=False):
        dir_vectors = {}
        for key in self.storage.keys():
            if key not in self.current_mapped_ids:
                continue
            dir_vector, dir_type = self.get_single_dir_vector(key)
            if dir_type is not HeadingType.UNAVAILABLE:
                dir_vectors[key] = dir_vector
            else:
                if cleanup:
                    # print(f"Deleting key {key}")
                    self.current_mapped_ids.remove(key)
                    del self.ids_mapped_from_input[key]
            if not cleanup:
                # TODO: Check this
                if self.storage[key][-1].flag == MeasType.STATIC and self.storage[key][-1].heading_type not in [
                    HeadingType.EMPTY, HeadingType.UNAVAILABLE]:
                    dir_vectors[key] = self.storage[key][-1].heading
        return dir_vectors

    def get_dir_vectors_sorted(self):
        vectors = self.get_dir_vectors()
        return [vectors[_id] for _id in self.current_mapped_ids]

    def get_boxes_sorted(self, return_ids=False, return_history=False):
        if return_history:
            return self.get_box_and_hdg_history()
        else:
            boxes = self.get_last_boxes()
        if return_ids:
            return [boxes[_id] for _id in self.current_mapped_ids], self.current_mapped_ids
        else:
            return [boxes[_id] for _id in self.current_mapped_ids]

    def add_additional_information_sorted(self, additional_data: Dict[int, str],
                                          data_type: MetadataType = MetadataType.REFERENCE):
        if data_type != MetadataType.REFERENCE:
            raise ValueError(f"{data_type} is not a valid metadata type!")
        for _id, _ref in additional_data.items():
            if _id in self.storage.keys():
                self.storage[_id][-1].metadata = MeasMetadata(_ref)
                self.storage[_id][-1].metadata_type = data_type
            else:
                raise ValueError(f"ID {_id} is not a valid track ID!")

    def get_additional_information_sorted(self, data_type: MetadataType = MetadataType.REFERENCE):
        if data_type != MetadataType.REFERENCE:
            raise ValueError(f"{data_type} is not a valid metadata type!")
        return {_id: [elem.metadata.reference_point if elem.metadata_type == data_type else None
                      for elem in self.storage[_id]] for _id in
                self.current_mapped_ids}

    def get_additional_information_single(self, key: int, data_type: MetadataType = MetadataType.REFERENCE):
        if data_type != MetadataType.REFERENCE:
            raise ValueError(f"{data_type} is not a valid metadata type!")
        if key in self.current_mapped_ids:
            return [elem.metadata.reference_point if elem.metadata_type == data_type else None for elem in
                    self.storage[key]]
        else:
            raise ValueError(f"Object ID {key} is currently not mapped!")

    def filter_list(self, input_list):
        return [input_list[_id] for _id in self.current_mapped_ids]

    def get_last_boxes(self):
        boxes = {}
        for key in self.storage.keys():
            boxes[key] = self.storage[key][-1].box
        return boxes

    def get_box_and_hdg_history(self):
        boxes = {}
        for key in self.current_mapped_ids:
            mask_reg = self.get_flag_mask_for_key(key, MeasType.REG)
            mask_hdg = np.array([elem not in [HeadingType.EMPTY, HeadingType.UNAVAILABLE] for elem in
                                 get_heading_types(self.storage[key])])
            mask = (mask_reg & mask_hdg)
            boxes[key] = (get_boxes(self.storage[key]), get_headings(self.storage[key]), mask)
        return boxes

    def get_classes(self, use_history=False):
        classes = {}
        for key in self.current_mapped_ids:
            if use_history:
                class_history = [meas.classification for meas in self.storage[key]]
                ped_count = class_history.count(Classification.PEDESTRIAN)
                bike_count = class_history.count(Classification.BICYCLE)
                classes[key] = Classification.PEDESTRIAN
            else:
                classes[key] = self.storage[key][-1].classification
        return [classes[_id] for _id in self.current_mapped_ids]

    def get_box_and_hdg_history_single(self, key):
        if key in self.current_mapped_ids:
            mask_reg = self.get_flag_mask_for_key(key, MeasType.REG)
            mask_hdg = np.array([elem not in [HeadingType.EMPTY, HeadingType.UNAVAILABLE] for elem in
                                 get_heading_types(self.storage[key])])
            mask = (mask_reg & mask_hdg)
            return get_boxes(self.storage[key]), get_headings(self.storage[key]), mask
        else:
            raise ValueError(f"Object ID {key} is currently not mapped!")

    def get_flag_mask_for_key(self, key, type_):
        return np.array([f == type_ for f in get_flags(self.storage[key])])

    def trim_history(self):
        if self.qs is None or self.qs < 2:
            return
        for key in self.storage.keys():
            if len(self.storage[key]) > self.qs:
                valid_meas_num = self.get_flag_mask_for_key(key, MeasType.INIT).sum() \
                                 + self.get_flag_mask_for_key(key, MeasType.REG).sum()
                if valid_meas_num < 3:
                    while len(self.storage[key]) > self.qs:
                        deletion_candidates = self.get_flag_mask_for_key(key, MeasType.STATIC) \
                                              | self.get_flag_mask_for_key(key, MeasType.COPY)
                        index = np.nonzero(deletion_candidates)[0][0]
                        del self.storage[key][index]
                self.storage[key] = self.storage[key][-self.qs:]

    def remove_unlikely_tracks(self):
        keys = list(self.storage.keys())
        for key in keys:
            if len(self.storage[key]) >= self.decay_after and np.all(
                    np.array([flag == MeasType.COPY for flag in get_flags(self.storage[key])[-self.decay_after:]])):
                if self.store_tracks:
                    self.track_storage.append(
                        {"boxes": get_boxes(self.storage[key]), "flags": get_flags(self.storage[key]),
                         "timestamps": get_timestamps(self.storage[key])})
                del self.storage[key]
                del self.track_metadata[key]
                if self.current_mapped_ids is not None and key in self.current_mapped_ids:
                    self.current_mapped_ids.remove(key)
                    del self.ids_mapped_from_input[key]

    def get_sanitized_tracks(self):
        ret_tracks = []
        for i in range(len(self.track_storage)):
            flags = self.track_storage[i]["flags"]
            mask = np.array([f != MeasType.COPY for f in flags])
            longest_copy_seq = 0
            for f in flags:
                if f == MeasType.COPY:
                    longest_copy_seq += 1
                else:
                    longest_copy_seq = 0
                if longest_copy_seq >= self.decay_after:
                    break
            if longest_copy_seq >= self.decay_after:
                timestamps = np.array(self.track_storage[i]["timestamps"], dtype=int)[mask]
                boxes = np.array(self.track_storage[i]["boxes"])[mask, :]
                ret_tracks.append((timestamps, boxes))
        return ret_tracks
    
    def get_estimates(self) -> List[Estimate]:
        vectors = self.get_dir_vectors()
        boxes = self.get_last_boxes()
        self.force_update_filtered_dirs()
        estimates = []
        for idx in self.current_mapped_ids:
            dir_vector = vectors[idx]
            # TODO: Check if filtering two times is a bug
            dir_vector_filtered = self.get_filter_dir(idx)
            if isinstance(dir_vector_filtered, tuple):
                dir_vector_filtered = dir_vector_filtered[0]
            full_box = boxes[idx]
            classification = self.storage[idx][-1].classification
            displacement = (np.linalg.norm(self.track_metadata[idx].total_displacement), self.track_metadata[idx].length)
            estimates.append(Estimate(full_box=full_box, dir_vector=dir_vector, object_id=idx, classification=classification, displacement=displacement, dir_vector_filtered=dir_vector_filtered))
        return estimates
        


def filter_targets(targets, kwargs):
    targets_np, indices = np.unique(targets, axis=0, return_index=True)
    mask = (np.abs(targets_np[:, 3]) > 0.2) & (np.abs(targets_np[:, 3]) < 20)
    clutter = targets_np[~mask]
    if mask.sum() > 0:
        targets_np = targets_np[mask]
        indices = indices[mask]
        target_fit = np.copy(targets_np[:, [0, 1, 3]])
        target_fit[:, 2] *= kwargs["r_factor"]
        return target_fit, indices, clutter
    else:
        return None, None, targets_np


def color_tuple_to_string(color):
    r = int(color[0] * 255)
    g = int(color[1] * 255)
    b = int(color[2] * 255)
    a = int(color[3] * 255) if len(color) == 4 else 255
    return f"{r:x}{g:x}{b:x}{a:x}"


class DoubleBufferedPointBuffer:

    def __init__(self, p=None):
        self.points = {}
        self.batches = []
        self.lines = []
        self.line_batches = []
        self.p = p

    def push_points(self, points, color=(1.0, 1.0, 1.0, 1.0)):
        if type(color) != str:
            color = color_tuple_to_string(color)
        if points.ndim == 1:
            points = points.reshape((1, -1))
        if color in self.points.keys():
            self.points[color].append(points)
        else:
            self.points[color] = [points]

    def push_lines(self, lines):
        self.lines.append(lines)

    def estimate_box_lines(self, cluster):
        self.lines.append(fit_aligned_bounding_box(cluster, draw_commands=True))
        # self.lines.append(fit_box(cluster, self.p, draw_commands=True))

    def update_points(self, implot=False, clear=True):
        batches = []
        for key in self.points.keys():
            color = tuple(int(key[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))
            points = np.concatenate(self.points[key], axis=0)
            if implot:
                batches.append((points.T[0], points.T[1], color))
            else:
                batches.append((points, color))
        self.batches = batches
        if clear:
            self.points = {}

    def update_lines(self, clear=True):
        self.line_batches = self.lines
        if clear:
            self.lines = []


class TargetTracker:

    def __init__(self, filtering_function=None, queue_size=20, decay_after=5, aggregate_last_n_timestaps=3,
                 store_tracks=False, use_3d=False):
        self.qs = queue_size
        self.decay_after = decay_after
        self.depth = aggregate_last_n_timestaps
        self.store_tracks = store_tracks
        self.target_storage = deque()
        self.time_storage = deque()
        self.track_history = {}
        self.flags = {}
        self.timestamps = {}
        self.track_storage = []
        self.added_meas_in_cur_step = True
        self.current_mapped_ids = None
        if filtering_function is None:
            self.filtering_function = filter_targets
        else:
            self.filtering_function = filtering_function
        self.kwargs = {"r_factor": 0.4, "dbscan_eps": 2.5, "dbscan_min_cluster_size": 3, "assoc_thr": 10.0}
        self.use_3d = use_3d
        self.ci = 3 if self.use_3d else 2
        self.cur_index = 0

    def incr_index(self):
        self.cur_index = (self.cur_index + 1) % 1024
        while self.cur_index in self.track_history.keys():
            self.cur_index = (self.cur_index + 1) % 1024

    def unlock_for_cur_step(self):
        self.added_meas_in_cur_step = False
        self.current_mapped_ids = None

    def extrapolate_track(self, i):
        def estimate_displacement(a1, a2):
            a1 = np.copy(a1) if a1.ndim == 2 else np.copy(a1).reshape((1, -1))
            a2 = np.copy(a2) if a2.ndim == 2 else np.copy(a2).reshape((1, -1))
            cost_mat = np.zeros((a1.shape[0], a2.shape[0]))
            if a1.shape[0] == 1 and a2.shape[0] == 1:
                return a2[0, :self.ci] - a1[0, :self.ci]
            for k in range(a1.shape[0]):
                for j in range(a2.shape[0]):
                    cost_mat[k, j] = np.linalg.norm(a1[k, :self.ci] - a2[j, :self.ci])
            row_ids, col_ids = linear_sum_assignment(cost_mat)
            dists = []
            for k, j in zip(row_ids, col_ids):
                dists.append(a2[j, :self.ci] - a1[k, :self.ci])
            if len(dists) == 1:
                return dists[0]
            return np.concatenate(dists, axis=0).mean(axis=0)

        if i in self.track_history.keys():
            n = len(self.track_history[i])
            if n == 1:
                return self.track_history[i][-1]
            # elif n == 2:
            else:
                displacement = estimate_displacement(self.track_history[i][-2], self.track_history[i][-1]).reshape(
                    (1, -1))
                base = np.zeros((1, 10))
                base[0, :self.ci] = displacement[0]
                return self.track_history[i][-1] + base
            # else:
            #     d1 = estimate_displacement(self.track_history[i][-2], self.track_history[i][-1]).reshape((1, -1))
            #     d2 = estimate_displacement(self.track_history[i][-3], self.track_history[i][-2]).reshape((1, -1))
            #     base1 = np.zeros((1, 10))
            #     base2 = np.zeros((1, 10))
            #     base1[0, :2] = d1[0]
            #     base2[0, :2] = d2[0]
            #     return self.track_history[i][-1] + base1 + (base1 - base2)

    # TODO: Check copy paste
    def get_flag_mask_for_key(self, key, type_):
        return np.array([f == type_ for f in self.flags[key]])

    # TODO: Check copy paste
    def trim_history(self):
        if self.store_tracks or self.qs is None or self.qs < 2:
            return
        for key in self.track_history.keys():
            if len(self.track_history[key]) > self.qs:
                valid_meas_num = self.get_flag_mask_for_key(key, MeasType.INIT).sum() \
                                 + self.get_flag_mask_for_key(key, MeasType.REG).sum()
                if valid_meas_num < 3:
                    while len(self.track_history[key]) > self.qs:
                        deletion_candidates = self.get_flag_mask_for_key(key, MeasType.STATIC) \
                                              | self.get_flag_mask_for_key(key, MeasType.COPY)
                        if deletion_candidates.sum() == 0:
                            break
                        index = np.nonzero(deletion_candidates)[0][0]
                        del self.track_history[key][index]
                        del self.flags[key][index]
                        del self.timestamps[key][index]
                self.track_history[key] = self.track_history[key][-self.qs:]
                self.flags[key] = self.flags[key][-self.qs:]
                self.timestamps[key] = self.timestamps[key][-self.qs:]

    # TODO: Check copy paste
    def remove_unlikely_tracks(self):
        keys = list(self.track_history.keys())
        for key in keys:
            if len(self.track_history[key]) >= self.decay_after and np.all(
                    np.array([flag == MeasType.COPY for flag in self.flags[key][-self.decay_after:]])):
                if self.store_tracks:
                    self.track_storage.append({"targets": self.track_history[key], "flags": self.flags[key],
                                               "timestamps": self.timestamps[key]})
                del self.track_history[key]
                del self.flags[key]
                del self.timestamps[key]

    def add_targets(self, targets, t, debug_buffer=None):
        if self.added_meas_in_cur_step:
            return

        self.target_storage.append(targets)
        self.time_storage.append(t)

        accumulated_targets = np.concatenate(self.target_storage, axis=0)
        last_old_index = len(accumulated_targets) - len(targets)

        filtered_targets, indices, clutter = self.filtering_function(accumulated_targets, self.kwargs)

        if len(clutter) > 0 and debug_buffer is not None:
            debug_buffer.push_points(clutter[:, :2], color="808080FF")

        # return_clusters = []

        if filtered_targets is not None:

            db = DBSCAN(eps=self.kwargs["dbscan_eps"], min_samples=self.kwargs["dbscan_min_cluster_size"]).fit(
                filtered_targets)
            labels = db.labels_
            unique_labels = set(labels)
            if -1 in unique_labels:
                if debug_buffer is not None:
                    debug_buffer.push_points(filtered_targets[labels == -1, :2], color="B0C4DEAA")
                unique_labels.remove(-1)

            num_inputs = len(unique_labels)

            unique_labels = list(unique_labels)

            for i, label in enumerate(unique_labels):
                cluster_targets = filtered_targets[labels == label, :self.ci]
                if debug_buffer is not None:
                    debug_buffer.push_points(cluster_targets[:, :self.ci], color=plt.cm.Wistia(i / num_inputs))
                if debug_buffer is not None and len(cluster_targets) > 2:
                    debug_buffer.estimate_box_lines(cluster_targets)
                # return_clusters.append((cluster_targets, plt.cm.cool(i / num_inputs)[:3]))

        else:

            num_inputs = 0
            labels = []
            unique_labels = []

        cluster_centers = []
        cluster_radii = []
        if num_inputs > 0:
            for label in unique_labels:
                cluster_targets = filtered_targets[labels == label, :self.ci]
                cluster_centers.append(cluster_targets.mean(axis=0))
                target_dists = np.linalg.norm((cluster_targets - cluster_centers[-1].reshape((-1, self.ci))), axis=1)
                cluster_radii.append(target_dists.max())

        num_tracks = len(list(self.track_history.keys()))

        open_track_ids = list(self.track_history.keys())

        if num_tracks > 0:
            if num_inputs > 0:
                cost_mat = np.zeros((num_tracks, num_inputs))
                for i, label in enumerate(unique_labels):
                    for j, track_id in enumerate(open_track_ids):
                        cost_mat[j, i] = np.linalg.norm(
                            cluster_centers[i] - self.track_history[track_id][-1][:, :self.ci].mean(axis=0))

                # Row: indices of objects of previous frame, col: indices of objects of current frame
                row_ids, col_ids = linear_sum_assignment(cost_mat)

                assoc_labels = []
                assoc_tracks = []

                for row, col in zip(row_ids, col_ids):
                    cost = cost_mat[row, col]
                    # target_sample_speed = filtered_targets[labels == unique_labels[col]][0][3]
                    if cost >= self.kwargs["assoc_thr"]:
                        continue
                    selected_indices = indices[labels == unique_labels[col]]
                    selected_indices = selected_indices[selected_indices >= last_old_index]
                    selected_count = (selected_indices >= last_old_index).sum()
                    if selected_count > 0:
                        idx = open_track_ids[row]
                        selected_targets = accumulated_targets[selected_indices].reshape((selected_count, -1))
                        self.track_history[idx].append(selected_targets)
                        self.flags[idx].append(MeasType.REG)
                        self.timestamps[idx].append(t)
                        assoc_labels.append(unique_labels[col])
                        assoc_tracks.append(idx)
                        # if debug_buffer is not None and len(selected_targets) > 2:
                        #     debug_buffer.estimate_box_lines(selected_targets)

                open_track_ids = list(set(open_track_ids) - set(assoc_tracks))
                unique_labels = list(set(unique_labels) - set(assoc_labels))
            else:
                pass
        else:
            if num_inputs > 0:

                for label in unique_labels:
                    selected_indices = indices[labels == label]
                    selected_indices = selected_indices[selected_indices >= last_old_index]
                    selected_count = (selected_indices >= last_old_index).sum()
                    if selected_count > 0:
                        selected_targets = accumulated_targets[selected_indices].reshape((selected_count, -1))
                        self.track_history[self.cur_index] = [selected_targets]
                        self.flags[self.cur_index] = [MeasType.INIT]
                        self.timestamps[self.cur_index] = [t]
                        # if debug_buffer is not None and len(selected_targets) > 2:
                        #     debug_buffer.estimate_box_lines(selected_targets)
                        self.incr_index()

        for key in open_track_ids:
            self.track_history[key].append(self.extrapolate_track(key))
            self.flags[key].append(MeasType.COPY)
            self.timestamps[key].append(t)

        if len(unique_labels) > 0:

            for label in unique_labels:
                selected_indices = indices[labels == label]
                selected_indices = selected_indices[selected_indices >= last_old_index]
                selected_count = (selected_indices >= last_old_index).sum()
                if selected_count > 0:
                    selected_targets = accumulated_targets[selected_indices].reshape((selected_count, -1))
                    self.track_history[self.cur_index] = [selected_targets]
                    self.flags[self.cur_index] = [MeasType.INIT]
                    self.timestamps[self.cur_index] = [t]
                    # if debug_buffer is not None and len(selected_targets) > 2:
                    #     debug_buffer.estimate_box_lines(selected_targets)
                    self.incr_index()

        # TODO: Is self.time_storage needed?
        if len(self.target_storage) > self.depth:
            self.target_storage.popleft()
            self.time_storage.popleft()

        self.remove_unlikely_tracks()

        self.trim_history()

        self.added_meas_in_cur_step = True

    def get_dirty_track_means(self):
        track_means = []
        for track in self.track_history.values():
            buffer = []
            for targets in track:
                buffer.append(targets.mean(axis=0)[:self.ci])
            track_means.append(np.array(buffer))
        return track_means

    def get_clusters(self):
        clusters = []
        for uid, track in self.track_history.items():
            if self.flags[uid][-1] == MeasType.REG:
                clusters.append((uid, track[-1]))
        return clusters

    def get_sanitized_tracks(self):
        ret_tracks = []
        for i in range(len(self.track_storage)):
            flags = self.track_storage[i]["flags"]
            mask = np.array([f != MeasType.COPY for f in flags])
            longest_copy_seq = 0
            for f in flags:
                if f == MeasType.COPY:
                    longest_copy_seq += 1
                else:
                    longest_copy_seq = 0
                if longest_copy_seq >= self.decay_after:
                    break
            if longest_copy_seq >= self.decay_after:
                timestamps = np.array(self.track_storage[i]["timestamps"], dtype=int)[mask]
                targets = [self.track_storage[i]["targets"][j] for j, flag in enumerate(mask) if flag]
                ret_tracks.append((timestamps, targets))
        return ret_tracks
