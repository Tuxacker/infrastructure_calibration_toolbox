from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..trackers import BoxTracker
from ..utils.bbox import not_at_edge
from ..utils.enums import Classification
from ..utils.misc import copy_to_2d_floats
from ..utils.files import load_file, load_from_cache
from ..utils.time import find_matches


# The calibration data source
class SourceType(Enum):
    INTERSECTION_SPU = 1
    CARLA = 2
    SIL = 3


# Correspondence container
@dataclass
class Correspondence:
    t: np.ndarray
    gnss: np.ndarray
    box: np.ndarray
    opt: Optional[np.ndarray]


# Hypothesis container
@dataclass
class Hypothesis:
    _id: int
    data_id: int
    r_vec: np.ndarray
    t: np.ndarray
    inlier_ids: np.ndarray


# Configuration parameters
@dataclass
class CalibrationConfig:
    _type: SourceType
    camera_metadata_path: str = ""
    maximal_distance_in_heatmap: float = 0.0
    heatmap_sparsity: int = 0
    maximal_median_projection_error: float = 0.0
    maximal_camera_distance_to_driven_path: float = 0.0
    maximal_sensor_range: float = 0.0
    filter_negative_heights: bool = False
    minimal_sequence_length: int = 0
    minimal_image_track_length: float = 0.0
    minimal_world_track_length: float = 0.0
    use_ransac: bool = False
    maximal_projection_error: int = 0
    maximal_ransac_iterations: int = 0
    tracker_decay_after: int = 0
    use_yolov5_labels: bool = False
    outlier_score_threshold: float = 0.0
    overlap_score_threshold: float = 0.0
    rotational_similarity_threshold: float = 0.0
    dbscan_eps: float = 0.0
    dbscan_min_samples: int = 0
    vehicle_length: float = 0.0
    vehicle_width: float = 0.0
    vehicle_height: float = 0.0
    imu_x: float = 0.0
    imu_y: float = 0.0
    imu_z: float = 0.0
    minimal_distance_to_image_edge: float = 0.0
    lifting_threshold: float = 0.0
    matching_threshold: float = 0.01
    verbose: bool = False

    camera_width: int = 0
    camera_height: int = 0
    k_matrix: np.ndarray = np.eye(3)
    distortion_coefficients: Optional[np.ndarray] = None

    # SPU only
    grid_tile_size: Optional[float] = None
    spu_id: Optional[int] = None
    cam_id: Optional[int] = None
    camera_name: Optional[str] = None

    # CARLA only
    camera_fov: Optional[float] = None
    camera_x: Optional[float] = None
    camera_y: Optional[float] = None
    camera_z: Optional[float] = None
    camera_pitch: Optional[float] = None
    camera_yaw: Optional[float] = None

    # SIL only
    offset: Optional[np.ndarray] = None

    # ground truth, CARLA and SIL only
    gt: Optional[np.ndarray] = None

    @staticmethod
    def load(cls, config_path: Path, **kwargs):
        config_dict = load_file(config_path)
        source_type = SourceType[config_dict["type"].upper()]
        config = CalibrationConfig(source_type)

        for k, v in config_dict.items():
            if k == "type":
                continue
            elif k == "offset":
                config.__setattr__(k, np.array(v))
                continue
            config.__setattr__(k, v)

        base_platform_path = Path(__file__).parents[2] / "config"


        if "radar" in kwargs:
            return config

        if source_type == SourceType.INTERSECTION_SPU:
            
            from .integrations.spu import load_camera_parameters

            W, H, K = load_camera_parameters(base_platform_path / Path(config_dict["camera_metadata_path"]), **kwargs)

            config.camera_width = W
            config.camera_height = H
            config.k_matrix = K
            config.distortion_coefficients = None
            if "spu_id" in kwargs and "cam_id" in kwargs:
                config.spu_id = kwargs["spu_id"]
                config.cam_id = kwargs["cam_id"]
            else:
                config.camera_name = kwargs["camera_name"]
            
        elif source_type == SourceType.SIL:

            from .integrations.sil import load_camera_parameters

            W, H, K, gt = load_camera_parameters(base_platform_path / Path(config_dict["camera_metadata_path"]), **kwargs)

            config.camera_width = W
            config.camera_height = H
            config.k_matrix = K
            config.distortion_coefficients = None
            config.gt = gt

        elif source_type == SourceType.CARLA:

            from .integrations.sil import load_camera_parameters

            K, gt = load_camera_parameters(config_dict, **kwargs)

            config.k_matrix = K
            config.distortion_coefficients = None
            config.gt = gt

        else:
            raise NotImplementedError()
        
        return config
    

# Input data container
@dataclass
class CalibrationData:
    data: List[Correspondence]
    gnss_data: np.ndarray
    config: CalibrationConfig
    heatmap: np.ndarray
    pair_count: int

    def __init__(self, associated_tracks: List[Correspondence], gnss_data: np.ndarray, config: CalibrationConfig):
        self.data = associated_tracks
        self.gnss_data = gnss_data
        self.config = config
        self.heatmap = self.calculate_heatmap()
        self.pair_count = len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def calculate_heatmap(self):
        concatenated_box_points = np.concatenate(
            [copy_to_2d_floats(track_pair.gnss).astype(int) for track_pair in self.data], axis=0).reshape(-1, 2)
        return np.unique((concatenated_box_points // self.config.heatmap_sparsity) * self.config.heatmap_sparsity,
                         axis=0)

    def get_heatmap(self):
        if self.heatmap is None:
            self.heatmap = self.calculate_heatmap()
            return self.heatmap
        else:
            return self.heatmap
        
    @classmethod
    def load_data(cls, _imu_data_path: Path, _detections_data_path: Path, _config: CalibrationConfig, _offset: Union[int, float] = 0):
        if _config.verbose:
            from tqdm import tqdm

        def load_data_impl(imu_data_path: Path, detections_data_path: Path, config: CalibrationConfig, offset: Union[int, float]=0):
            detections = load_from_cache(detections_data_path, verbose=config.verbose)
            positions = load_from_cache(imu_data_path, verbose=config.verbose)

            detections = {int(k): v for k, v in detections.items()}

            # Subtract possible time offset between data sources, e.g., TAI to GPS
            positions = {int(k) - offset: v for k, v in positions.items()}

            if config.verbose:
                print("IMU first timestamp:", list(positions.keys())[0])
                print("DET first timestamp:", list(detections.keys())[0])

            # Match detections to IMU measurements
            matches = find_matches(positions, detections, matching_thr=config.matching_threshold, verbose=config.verbose)

            tracker = BoxTracker(class_config={}, queue_size=None, store_tracks=True)
            tracker.decay_after = config.tracker_decay_after

            def not_at_edge_with_params(box):
                return not_at_edge(box, config.camera_width, config.camera_height, config.minimal_distance_to_image_edge)

            match_iter = tqdm(matches) if config.verbose else matches

            for match in match_iter:
                labels, _, t, _ = match

                # Filter out any objects that are not cars
                if config._type == SourceType.INTERSECTION_SPU:
                    labels = np.array([arr for arr in labels if 1 < int(arr[-1]) == int(Classification.CAR)])
                else:
                    labels = np.array(labels)
                
                # Filter out potentially incomplete bounding boxes
                labels = labels[np.array([not_at_edge_with_params(l) for l in labels], dtype=bool)]

                if labels.shape[0] > 0:
                    tracker.unlock_for_cur_step()
                    tracker.add_measurements(labels, int(t * 1e9))

            tracks = tracker.get_sanitized_tracks()

            track_pairs = []

            # TODO: Cleanup of different formats

            matched_t = np.array([int(m[2] * 1e9) for m in matches], dtype=float)

            if config._type == SourceType.CARLA:
                matched_gnss = np.array([m[1]["p"] for m in matches], dtype=float)
            else:
                matched_gnss = np.array([m[1] for m in matches], dtype=float)

            if config._type == SourceType.SIL:
                matched_gnss[:, :3] -= config.offset[None, :]
            for track in tracks:
                t, boxes = track

                matched_indices = np.array([np.where(np.abs(matched_t - stamp) < 1e5)[0] for stamp in t],
                                           dtype=int).ravel()
                gnss_pos = matched_gnss[matched_indices]

                if gnss_pos.shape[0] != boxes.shape[0]:
                    continue
                assert gnss_pos.shape[0] == boxes.shape[0]
                if config._type == SourceType.CARLA:
                    opt = np.array([m[1]["b"] for m in matches], dtype=float)
                else:
                    opt = np.array([m[1] for m in matches], dtype=float)
                track_pairs.append(Correspondence(t, gnss_pos, boxes, opt))

            if config._type == SourceType.CARLA:
                gnss_data = np.array([positions[stamp]["p"] for stamp in sorted(positions.keys())])
            else:
                gnss_data = np.array([positions[stamp] for stamp in sorted(positions.keys())])

            if config._type == SourceType.SIL:
                gnss_data[:, :3] -= config.offset[None, :]

            return CalibrationData(track_pairs, gnss_data, config)
        
        return load_from_cache(_detections_data_path, load_data_impl(_imu_data_path, _detections_data_path, _config, _offset), "_track_cache", verbose=_config.verbose)
    
    @classmethod
    def load_data_direct(cls, positions: dict, detections: dict, config: CalibrationConfig, offset: Union[int, float] = 0):
        if config.verbose:
            from tqdm import tqdm
        
        # Subtract possible time offset between data sources, e.g., TAI to GPS
        positions = {k - offset: v for k, v in positions.items()}

        if config.verbose:
            print("IMU first timestamp:", list(positions.keys())[0])
            print("DET first timestamp:", list(detections.keys())[0])

        # Match detections to IMU measurements
        matches = find_matches(positions, detections, matching_thr=config.matching_threshold, verbose=config.verbose)

        tracker = BoxTracker(class_config={}, queue_size=None, store_tracks=True)
        tracker.decay_after = config.tracker_decay_after

        def not_at_edge_with_params(box):
            return not_at_edge(box, config.camera_width, config.camera_height, config.minimal_distance_to_image_edge)

        match_iter = tqdm(matches) if config.verbose else matches

        for match in match_iter:
            labels, _, t, _ = match

            # Filter out any objects that are not cars
            labels = np.array([arr for arr in labels if 1 < int(arr[-1]) == int(Classification.CAR)])
            
            # Filter out potentially incomplete bounding boxes
            labels = labels[np.array([not_at_edge_with_params(l) for l in labels], dtype=bool)]

            if labels.shape[0] > 0:
                tracker.unlock_for_cur_step()
                tracker.add_measurements(labels, int(t * 1e9))

        tracks = tracker.get_sanitized_tracks()

        track_pairs = []

        # TODO: Cleanup of different formats

        matched_t = np.array([int(m[2] * 1e9) for m in matches], dtype=float)

        matched_gnss = np.array([m[1] for m in matches], dtype=float)

        for track in tracks:
            t, boxes = track

            matched_indices = np.array([np.where(np.abs(matched_t - stamp) < 1e5)[0] for stamp in t],
                                        dtype=int).ravel()
            gnss_pos = matched_gnss[matched_indices]

            if gnss_pos.shape[0] != boxes.shape[0]:
                continue
            assert gnss_pos.shape[0] == boxes.shape[0]
            opt = np.array([m[1] for m in matches], dtype=float)
            track_pairs.append(Correspondence(t, gnss_pos, boxes, opt))

        gnss_data = np.array([positions[stamp] for stamp in sorted(positions.keys())])

        return CalibrationData(track_pairs, gnss_data, config)
