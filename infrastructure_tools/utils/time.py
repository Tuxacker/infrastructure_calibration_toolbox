from datetime import datetime

import numpy as np

from .enums import TimestampType

def get_timestamp_type(timestamp):
    try:
        _ = datetime.utcfromtimestamp(int(timestamp))
        return TimestampType.SEC
    except OSError:
        return TimestampType.NANOSEC


def find_matches(positions, detections, matching_thr, verbose=False):
    sorted_keys_p = np.array(sorted(positions.keys()))
    sorted_keys_d = np.array(sorted(detections.keys()))
    sorted_keys_positions = np.copy(sorted_keys_p)
    sorted_keys_detections = np.copy(sorted_keys_d)

    timestamp_type_pos = get_timestamp_type(sorted_keys_positions[0])
    if verbose:
        print(f"Detected timestamp format: {timestamp_type_pos}")
    if timestamp_type_pos == TimestampType.NANOSEC:
        if verbose:
            print("Warning: Position timestamps are given in nanoseconds, converting to seconds")
        sorted_keys_positions = sorted_keys_positions.astype(float) / 1e9

    timestamp_type_det = get_timestamp_type(sorted_keys_detections[0])
    if timestamp_type_det == TimestampType.NANOSEC:
        if verbose:
            print("Warning: Detection timestamps are given in nanoseconds, converting to seconds")
        sorted_keys_detections = sorted_keys_detections.astype(float) / 1e9

    if verbose:
        positions_t_start = datetime.utcfromtimestamp(sorted_keys_positions[0]).strftime('%Y-%m-%d %H:%M:%S')
        positions_t_end = datetime.utcfromtimestamp(sorted_keys_positions[-1]).strftime('%Y-%m-%d %H:%M:%S')
        detections_t_start = datetime.utcfromtimestamp(sorted_keys_detections[0]).strftime('%Y-%m-%d %H:%M:%S')
        detections_t_end = datetime.utcfromtimestamp(sorted_keys_detections[-1]).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Localization time range : {positions_t_start} - {positions_t_end}")
        print(f"Detection time range : {detections_t_start} - {detections_t_end}")

    index_pos = 0
    index_det = 0

    def find_matching_pos_idx(t_det_in, idx_pos):
        while True:
            t_pos_in = sorted_keys_positions[idx_pos]
            t_pos_next = sorted_keys_positions[idx_pos + 1] if idx_pos < len(sorted_keys_positions) - 1 else 1e100
            if t_pos_in <= t_det_in <= t_pos_next:
                break
            idx_pos += 1
            if idx_pos >= len(sorted_keys_positions):
                return -1
        if abs(t_pos_next - t_det_in) < abs(t_pos_in - t_det_in):
            idx_pos += 1
        return idx_pos

    matches = []
    while index_det < len(sorted_keys_detections):
        t_det = sorted_keys_detections[index_det]
        index_pos = find_matching_pos_idx(t_det, index_pos)
        if index_pos < 0:
            index_det += 1
            index_pos = 0
            continue
        t_pos = sorted_keys_positions[index_pos]
        if abs(t_pos - t_det) > matching_thr:
            index_det += 1
            index_pos = 0
            continue

        if timestamp_type_pos == TimestampType.NANOSEC:
            t_pos_idx = int(t_pos * 1e9)
            t_pos_idx = sorted_keys_p[np.abs(sorted_keys_p - t_pos_idx).argmin()]
        else:
            t_pos_idx = t_pos

        if timestamp_type_det == TimestampType.NANOSEC:
            t_det_idx = int(t_det * 1e9)
            t_det_idx = sorted_keys_d[np.abs(sorted_keys_d - t_det_idx).argmin()]
        else:
            t_det_idx = t_det

        matches.append((detections[t_det_idx], positions[t_pos_idx], t_det, t_pos))
        index_det += 1

    if verbose:
        print(f"Found {len(matches)} matches")

    return matches