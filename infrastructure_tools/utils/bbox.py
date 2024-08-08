from typing import Union

import numpy as np
    
def xywh2xyxy(box):
    return np.array([box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2])

def xyxy2xywh(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])


def xyxy2corners(box):
    return np.array([[box[0], box[3]], [box[2], box[3]], [box[2], box[1]], [box[0], box[1]]])


def xywh2corners(box):
    return xyxy2corners(xywh2xyxy(box))

def not_at_edge(xywh: np.ndarray, w: Union[int, float], h: Union[int, float], threshold: Union[int, float]):
    xa, ya, xb, yb = xywh2xyxy(xywh)
    th = threshold
    if (th < xa < w - th) and (th < xb < w - th) and (th < ya < h - th) and (th < yb < h - th):
        return True
    else:
        return False