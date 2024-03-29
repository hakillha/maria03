import numpy as np

from tensorpack.utils import viz

from config import config as cfg

def draw_final_outputs(img, results, tags_on=True, bb_list_input=False):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    if tags_on:
        tags = []
        for r in results:
            tags.append(
                "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    else:
        tags = None
    if bb_list_input:
        boxes = np.asarray(results)
    else:
        boxes = np.asarray([r.box for r in results])
    ret = viz.draw_boxes(img, boxes, tags)

    return ret