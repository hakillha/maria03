import json
import numpy as np
import tqdm
from collections import namedtuple
from contextlib import ExitStack

from tensorpack import * # to import Callback
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

def detect_one_image(img, model_func, gt_boxes=None):
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
    if cfg.RE_ID.USE_DPM:
        return model_func(resized_img, gt_boxes)
    else:
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
    jsonable = False

    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        # if not using dpm dets the boxes would actually be file names
        for img, img_fname, boxes in df.get_data():
            result_list = []
            result_list.append(img_fname)

            if cfg.RE_ID.USE_DPM:
                fvs = detect_func(img, boxes)[0]
                
                bb_list = []
                fv_list = []
                for fv, box in zip(fvs, boxes):
                    bb_list.append(list(map(lambda x: round(float(x), 2), box)))
                    fv_list.append(fv.tolist())
                result_list.append(bb_list)
                result_list.append(fv_list)
            else:
                results = detect_func(img, None)

                bb_list = []
                label_list = []
                score_list = []
                fv_list = []
                for r in results:
                    box = r.box
                    bb_list.append(list(map(lambda x: round(float(x), 2), box)))
                    label_list.append(int(r.class_id))
                    score_list.append(round(float(r.score), 3))
                    fv_list.append(r.fv.tolist())
                result_list.append(bb_list)
                result_list.append(label_list)
                result_list.append(score_list)
                result_list.append(fv_list)
                
            # dump a dummy json here to check for validity
            if not jsonable:
                jsonable = jsonable_test(result_list)

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
        for fname, img, gt_boxes, gt_ids, orig_boxes in df.get_data():
            fvs = pred_func(img, gt_boxes)
            # print(fvs.shape)

            result_list = []
            fv_list = []
            id_list = []
            orig_bb_list = []
            for fv, gt_id in zip(fvs, gt_ids):
                id_list.append(int(gt_id))
                fv_list.append(fv.tolist())
                orig_bb_list.append(orig_boxes.tolist())
            result_list.append(fname)
            result_list.append(id_list)
            result_list.append(fv_list)
            result_list.append(orig_bb_list)
                
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


class EvalCallback(Callback):
    """
    A callback that runs COCO evaluation once a while.
    It supports multi-GPU evaluation if TRAINER=='replicated' and single-GPU evaluation if TRAINER=='horovod'
    """

    def __init__(self, in_names, out_names):
        self._in_names, self._out_names = in_names, out_names

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        # Use two predictor threads per GPU to get better throughput
        self.num_predictor = num_gpu * 2
        self.predictors = [self._build_predictor(k % num_gpu) for k in range(self.num_predictor)]
        self.dataflows = [get_eval_dataflow(shard=k, num_shards=self.num_predictor)
                          for k in range(self.num_predictor)]

    def _build_predictor(self, idx):
        graph_func = self.trainer.get_predictor(self._in_names, self._out_names, device=idx)
        return lambda img: detect_one_image(img, graph_func)

    def _before_train(self):
        num_eval = cfg.TRAIN.NUM_EVALS
        interval = max(self.trainer.max_epoch // (num_eval + 1), 1)
        self.epochs_to_eval = set([interval * k for k in range(1, num_eval + 1)])
        self.epochs_to_eval.add(self.trainer.max_epoch)
        # for debug purpose
        # self.epochs_to_eval.update([0, 1])
        if len(self.epochs_to_eval) < 15:
            logger.info("[EvalCallback] Will evaluate at epoch " + str(sorted(self.epochs_to_eval)))
        else:
            logger.info("[EvalCallback] Will evaluate every {} epochs".format(interval))

    def _eval(self):
        # with ThreadPoolExecutor(max_workers=self.num_predictor, thread_name_prefix='EvalWorker') as executor, \
        # compatible with python 3.5
        with ThreadPoolExecutor(max_workers=self.num_predictor) as executor, \
                tqdm.tqdm(total=sum([df.size() for df in self.dataflows])) as pbar:
            futures = []
            for dataflow, pred in zip(self.dataflows, self.predictors):
                futures.append(executor.submit(eval_output, dataflow, pred, pbar))
            all_results = list(itertools.chain(*[fut.result() for fut in futures]))

        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        # try:
        #     scores = print_evaluation_scores(output_file)
        #     for k, v in scores.items():
        #         self.trainer.monitors.put_scalar(k, v)
        # except Exception:
        #     logger.exception("Exception in COCO evaluation.")

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()