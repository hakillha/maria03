import json
import numpy as np
import tqdm
from collections import namedtuple
from contextlib import ExitStack

from tensorpack.utils.utils import get_tqdm_kwargs

from common import CustomResize, clip_boxes
from config import config as cfg

VIZ = False

if VIZ:
    import cv2
    import os
    import tensorpack.utils.viz as tpviz
    from viz import draw_final_outputs


DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'fv'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
"""

def jsonable_test(data, fname='dummy.json'):
    with open(fname, 'w') as f:
        json.dump(data, f)

    return True

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
    boxes, probs, labels, fv = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, fv)]
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

            result_list = []
            result_list.append(img_fname)

            bb_list = []
            label_list = []
            score_list = []
            fv_list = []
            for r in results:
                box = r.box
                bb_list.append(list(map(lambda x: round(float(x), 2), box)))
                label_list.append(int(r.class_id))
                score_list.append(round(float(r.score), 3))
                # print(len(r.fv.tolist()))
                fv_list.append(r.fv.tolist())
            result_list.append(bb_list)
            result_list.append(label_list)
            result_list.append(score_list)
            result_list.append(fv_list)
                
            # dump a dummy json here to check for validity
            all_results.append(result_list)
            tqdm_bar.update(1)
    return all_results

def query_eval_output(df, pred_func, tqdm_bar=None):
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
        for img, gt_boxes, gt_ids in df.get_data():
            fvs = pred_func(img, gt_boxes)
            # print(fvs.shape)

            result_list = []
            fv_list = []
            id_list = []
            for fv, gt_id in zip(fvs, gt_ids):
                fv_list.append(fv.tolist())
                id_list.append(int(gt_id))
            result_list.append(id_list)
            result_list.append(fv_list)
                
            all_results.append(result_list)
            tqdm_bar.update(1)
    return all_results

def classifier_eval_output(df, pred_func, tqdm_bar=None):
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
    jsonable = False
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for fname, img, orig_shape in df.get_data():
            bbs, probs = pred_func(img, orig_shape)

            if VIZ:
                input_file = os.path.join('/media/yingges/TOSHIBA EXT/datasets/re-ID/PRW-v16.04.20/frames', os.path.basename(fname).split('.')[0] + '.jpg')
                img = cv2.imread(input_file, cv2.IMREAD_COLOR)
                final = draw_final_outputs(img, bbs, tags_on=False, bb_list_input=True)
                viz = np.concatenate((img, final), axis=1)
                cv2.imwrite(os.path.basename(input_file), viz)

            result_list = []
            result_list.append(fname)
            bb_list = []
            prob_list = []
            for bb, prob in zip(bbs, probs):
                bb_list.append(list(map(lambda x: round(float(x), 2), bb)))
                prob_list.append(list(map(lambda x:round(float(x), 4), prob)))
            result_list.append(bb_list)
            result_list.append(prob_list)
            
            if not jsonable:
                jsonable = jsonable_test(result_list)

            all_results.append(result_list)
            tqdm_bar.update(1)
    return all_results