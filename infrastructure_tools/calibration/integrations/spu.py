from pathlib import Path
from yaml import safe_load

import numpy as np

def load_camera_parameters(config_path: Path, **kwargs):
    with open(config_path, "r") as f:
        metadata_string = f.readlines()
        metadata_string = "\n".join([s.replace("\t", "  ") for s in metadata_string])
        if "camera_name" in kwargs:
            camera_metadata = safe_load(metadata_string)[kwargs["platform_name"]]["devices"][kwargs["camera_name"]]
        else:
            camera_metadata = safe_load(metadata_string)[kwargs["platform_name"]]["devices"][f"camera_spu_{kwargs['spu_id']}_{kwargs['cam_id']}"]

    K = np.eye(3)

    if camera_metadata["type"] == "fisheye_camera":
        K[0, 0] = camera_metadata["output_focal_length"][0]
        K[1, 1] = camera_metadata["output_focal_length"][1]
        K[0, 2] = camera_metadata["output_optical_center"][0]
        K[1, 2] = camera_metadata["output_optical_center"][1]

        W, H = camera_metadata["output_resolution"]
    else:
        K[0, 0] = camera_metadata["focal_length"][0]
        K[1, 1] = camera_metadata["focal_length"][1]
        K[0, 2] = camera_metadata["optical_center"][0]
        K[1, 2] = camera_metadata["optical_center"][1]

        W, H = camera_metadata["resolution"]

    if "spu_id" in kwargs and "cam_id" in kwargs:

        if int(kwargs["spu_id"]) == 3:
            W, H = H, W
            fx, fy = K[0, 0], K[1, 1]
            K[0, 0], K[1, 1] = fy, fx
            cx, cy = K[0, 2], K[1, 2]
            K[0, 2], K[1, 2] = cy, H - cx

        if int(kwargs["spu_id"]) == 4 and int(kwargs["cam_id"]) == 2:
            W, H = H, W
            fx, fy = K[0, 0], K[1, 1]
            K[0, 0], K[1, 1] = fy, fx
            cx, cy = K[0, 2], K[1, 2]
            K[0, 2], K[1, 2] = W - cy, cx

    return W, H, K