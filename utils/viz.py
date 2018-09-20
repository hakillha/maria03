import cv2
import numpy as np

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB
from tensorpack.utils.rect import BoxBase, IntBox

from config import config as cfg


def draw_boxes(im, boxes, labels=None, color=None, line_thickness=2):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray or list[BoxBase]): If an ndarray,
            must be of shape Nx4 where the second dimension is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple (in range [0, 255]). By default will choose automatically.

    Returns:
        np.ndarray: a new image.
    """
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    if isinstance(boxes, list):
        arr = np.zeros((len(boxes), 4), dtype='int32')
        for idx, b in enumerate(boxes):
            assert isinstance(b, BoxBase), b
            arr[idx, :] = [int(b.x1), int(b.y1), int(b.x2), int(b.y2)]
        boxes = arr
    else:
        boxes = boxes.astype('int32')
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)    # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
        and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
        "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    COLOR = (218, 218, 218) if color is None else color
    COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2), dtype='int32')    # https://www.wikiwand.com/en/Color_difference
    COLOR_CANDIDATES = PALETTE_RGB[:, ::-1]
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]

        best_color = COLOR
        if labels is not None:
            label = labels[i]

            # find the best placement for the text
            ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
            top_left = [box[0] + 1, box[1] - 1.3 * lineh]
            if top_left[1] < 0:     # out of image
                top_left[1] = box[3] - 1.3 * lineh
                bottom_left[1] = box[3] - 0.3 * lineh
            textbox = IntBox(int(top_left[0]), int(top_left[1]),
                             int(top_left[0] + linew), int(top_left[1] + lineh))
            textbox.clip_by_shape(im.shape[:2])
            if color is None:
                # find the best color
                mean_color = textbox.roi(im).mean(axis=(0, 1))
                best_color_ind = (np.square(COLOR_CANDIDATES - mean_color) *
                                  COLOR_DIFF_WEIGHT).sum(axis=1).argmax()
                best_color = COLOR_CANDIDATES[best_color_ind].tolist()

            cv2.putText(im, label, (textbox.x1, textbox.y2),
                        FONT, FONT_SCALE, color=best_color, lineType=cv2.LINE_AA)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                      color=best_color, thickness=line_thickness)
    return im

def draw_final_outputs(img, results, 
                       tags_on=True, bb_list_input=False,
                       cls_list=None, gt_cls_list=None,
                       ids=None):
    """
    Args:
        results: [DetectionResult]
        cls_list, gt_cls_list: For classification eval
    """
    if len(results) == 0:
        return img

    if tags_on:
        tags = []
        if cls_list:
            for klass, gt_cls in zip(cls_list, gt_cls_list):
                tags.append("{}, {}".format(str(klass), str(gt_cls)))
        elif ids:
            for id_ in ids:
                tags.append('ID: {}'.format(str(int(id_))))
        else:
            for r in results:
                tags.append(
                    "{}, {:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    else:
        tags = None

    if bb_list_input:
        boxes = np.asarray(results)
    else:
        boxes = np.asarray([r.box for r in results])
    ret = draw_boxes(img, boxes, tags)

    return ret