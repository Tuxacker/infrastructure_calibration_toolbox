from pathlib import Path
from yaml import safe_load

import numpy as np

def load_camera_parameters(config_path: Path, **kwargs):
    with open(config_path, "r") as f:
        metadata_string = f.readlines()
        metadata_string = "\n".join([s.replace("\t", "  ") for s in metadata_string])
        camera_metadata = safe_load(metadata_string)["intersection_spu"]["devices"][kwargs["spu_id"]]

    K = np.eye(3)
    K[0, 0] = camera_metadata["focal_length"][0]
    K[1, 1] = camera_metadata["focal_length"][1]
    K[0, 2] = camera_metadata["optical_center"][0]
    K[1, 2] = camera_metadata["optical_center"][1]

    W, H = camera_metadata["resolution"]

    gt = np.linalg.inv(np.array(camera_metadata["alignment"]))

    return W, H, K, gt