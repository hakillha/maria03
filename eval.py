from collections import namedtuple
import numpy as np

from common import CustomResize, clip_boxes
from config import config as cfg


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
"""

def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels)

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels)]
    return results