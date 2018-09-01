import numpy as np
import tqdm
from collections import namedtuple
from contextlib import ExitStack

from tensorpack.utils.utils import get_tqdm_kwargs

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

def eval_output(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_fname in df.get_data():
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_name = cfg.DATA.CLASS_NAMES[r.class_id]

                res = {
                    'image_filename': img_fname,
                    'category_name': cat_name,
                    'bbox': list(map(lambda x: round(float(x), 2), box)),
                    'score': round(float(r.score), 3),
                }

                # also append segmentation to results
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results