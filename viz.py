import numpy as np

from tensorpack.utils import viz

from config import config as cfg

def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    tags = []
    for r in results:
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    boxes = np.asarray([r.box for r in results])
    ret = viz.draw_boxes(img, boxes, tags)

    return ret