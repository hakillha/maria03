import cv2
import copy
import numpy as np
import os
from os.path import join as pjoin
import scipy.io
import tqdm

from tensorpack.dataflow import (imgaug, MultiProcessMapDataZMQ,
                                 DataFromList, MapDataComponent,
                                 MapData)
from tensorpack.utils.argtools import memoized
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation

from common import (CustomResize, filter_boxes_inside_shape,
                    box_to_point8, point8_to_box,
                    DataFromListOfDict)
from config import config as cfg
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou


class MalformedData(BaseException):
    pass

class PRWDataset(object):

    def __init__(self, basedir):
        self._basedir = basedir
        self._imgdir = pjoin(basedir, 'frames')
        self._annodir = pjoin(basedir, 'annotations')
        if not cfg.DATA.CLASS_NAMES:
            cfg.DATA.CLASS_NAMES = ['BG', 'pedestrian']

    def load(self, split_set='train'):
        """
        Args:
            split_set: ['train', 'val']

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'id', 'file_name',
                and (if split_set is 'train') 'boxes', 'class', 'is_crowd'.
        """
        with timed_operation('Load Groundtruth Boxes...'):
            frame_list_mat = scipy.io.loadmat(pjoin(self._basedir, 'frame_' + split_set + '.mat'))
            frame_list = frame_list_mat['img_index_' + split_set]

            imgs = []
            imgs_without_fg = 0

            # each iteration only reads one file so it's faster
            for idx, frame in enumerate(frame_list):
                img = {}

                self._use_absolute_file_name(img, frame[0][0])

                if split_set == 'train':
                    if frame[0][0][1] == '6':
                        img['height'] = 576
                        img['width'] = 720
                    else:
                        img['height'] = 1080
                        img['width'] = 1920

                    anno_data = scipy.io.loadmat(pjoin(self._annodir, frame[0][0] + '.jpg.mat'))

                    if 'box_new' in anno_data:
                        gt_bb_array = anno_data['box_new']
                    elif 'anno_file' in anno_data:
                        gt_bb_array = anno_data['anno_file']
                    elif 'anno_previous' in anno_data:
                        gt_bb_array = anno_data['anno_previous']
                    else:
                        raise Exception(frame[0][0] + ' bounding boxes info missing!')
                    
                    img['boxes'] = []
                    for bb in gt_bb_array:
                        box = FloatBox(bb[1], bb[2], bb[1] + bb[3], bb[2] + bb[4])
                        box.clip_by_shape([img['height'], img['width']])
                        img['boxes'].append([box.x1, box.y1, box.x2, box.y2])
                    img['boxes'] = np.asarray(img['boxes'], dtype='float32')
                    img['class'] = np.ones(len(gt_bb_array))
                    img['re_id_class'] = np.asarray(gt_bb_array[:, 0], dtype='int32')
                    # 
                    if len(np.where(img['re_id_class'] == 932)[0]):
                        print('Last ID shifted.')
                    img['re_id_class'][img['re_id_class'] == 932] = 479


                    img['is_crowd'] = np.zeros(len(img['re_id_class']), dtype='int8')

                imgs.append(img)

            print('Number of images without identified pedestrians: {}.'.format(imgs_without_fg))
            return imgs

    def load_query(self):
        imgs = []
        with open(pjoin(self._basedir, 'query_info.txt'), 'r') as f:
            for line in f:
                img = {}
                line_list = line.split()
                self._use_absolute_file_name(img, line_list[5])

                if line_list[5][1] == '6':
                    img['height'] = 576
                    img['width'] = 720
                else:
                    img['height'] = 1080
                    img['width'] = 1920

                x1 = float(line_list[1])
                y1 = float(line_list[2])
                w = float(line_list[3])
                h = float(line_list[4])
                box = FloatBox(x1, y1, x1 + w, y1 + h)
                box.clip_by_shape([img['height'], img['width']])
                img['boxes'] = []
                img['boxes'].append([box.x1, box.y1, box.x2, box.y2])
                img['boxes'] = np.asarray(img['boxes'], dtype='float32')

                img['re_id_class'] = []
                img['re_id_class'].append(line_list[0])
                # ? this prob doesn't matter since unlabeled pedes are not considered in eval
                # img['re_id_class'] = np.asarray(img['re_id_class'], dtype='int32') + 1

                # we can remove this since it's only checked in dataflow processing
                # img['is_crowd'] = np.zeros(len(img['re_id_class']), dtype='int8')
                imgs.append(img)
            
        return imgs

    def load_dpm(self):
        frame_list = scipy.io.loadmat(pjoin(self._basedir, 'frame_test.mat'))['img_index_test']
        dpm_det_list = scipy.io.loadmat(pjoin(self._basedir, 'dpm_test.mat'))['dpm_test']
        imgs = []

        # each iteration only reads one file so it's faster
        for idx, (frame, frame_det_list) in enumerate(zip(frame_list, dpm_det_list)):
            img = {}

            self._use_absolute_file_name(img, frame[0][0])

            if frame[0][0][1] == '6':
                img['height'] = 576
                img['width'] = 720
            else:
                img['height'] = 1080
                img['width'] = 1920

            img['boxes'] = []
            for bb in frame_det_list[0]:
                box = FloatBox(bb[0], bb[1], bb[2], bb[3])
                if bb[2] < bb[0] or bb[3] < bb[1]:
                    print('Invalid detection results found!')
                    continue
                # assert bb[2] > bb[0], bb
                # assert bb[3] > bb[1], bb
                box.clip_by_shape([img['height'], img['width']])
                img['boxes'].append([box.x1, box.y1, box.x2, box.y2])
            img['boxes'] = np.asarray(img['boxes'], dtype='float32')

            imgs.append(img)


        return imgs

    def _use_absolute_file_name(self, img, file_name):
        """
        Change relative filename to abosolute file name.
        """

        img['file_name'] = pjoin(self._imgdir, file_name + '.jpg')
        assert os.path.isfile(img['file_name'])


@memoized
def get_all_anchors(stride=None, sizes=None):
    """
    Get all anchors in the largest possible image, shifted, floatbox
    Args:
        stride (int): the stride of anchors.
        sizes (tuple[int]): the sizes (sqrt area) of anchors

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    if stride is None:
        stride = cfg.RPN.ANCHOR_STRIDE
    if sizes is None:
        sizes = cfg.RPN.ANCHOR_SIZES
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(cfg.RPN.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    max_size = cfg.PREPROC.MAX_SIZE
    field_size = int(np.ceil(max_size / stride)) # receptive field?
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors

def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= cfg.RPN.POSITIVE_ANCHOR_THRESH] = 1
    anchor_labels[ious_max_per_anchor < cfg.RPN.NEGATIVE_ANCHOR_THRESH] = 0

    # We can label all non-ignore candidate boxes which overlap crowd as ignore
    # But detectron did not do this.
    # if crowd_boxes.size > 0:
    #     cand_inds = np.where(anchor_labels >= 0)[0]
    #     cand_anchors = anchors[cand_inds]
    #     ious = np_iou(cand_anchors, crowd_boxes)
    #     overlap_with_crowd = cand_inds[ious.max(axis=1) > cfg.RPN.CROWD_OVERLAP_THRES]
    #     anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(cfg.RPN.BATCH_PER_IM * cfg.RPN.FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Keep an image even if there is no foreground anchors
    # if len(fg_inds) == 0:
    #     raise MalformedData("No valid foreground for RPN!")

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg in this image, skip.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = cfg.RPN.BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == cfg.RPN.BATCH_PER_IM
    return anchor_labels, anchor_boxes

def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
        NA will be NUM_ANCHOR_SIZES x NUM_ANCHOR_RATIOS
    """
    boxes = boxes.copy()
    all_anchors = np.copy(get_all_anchors())
    # fHxfWxAx4 -> (-1, 4)
    featuremap_anchors_flatten = all_anchors.reshape((-1, 4))

    # only use anchors inside the image
    inside_ind, inside_anchors = filter_boxes_inside_shape(featuremap_anchors_flatten, im.shape[:2])
    # obtain anchor labels and their corresponding gt boxes
    anchor_labels, anchor_gt_boxes = get_anchor_labels(inside_anchors, boxes[is_crowd == 0], boxes[is_crowd == 1])

    # Fill them back to original size: fHxfWx1, fHxfWx4
    anchorH, anchorW = all_anchors.shape[:2]
    featuremap_labels = -np.ones((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR))
    featuremap_boxes = np.zeros((anchorH * anchorW * cfg.RPN.NUM_ANCHOR, 4), dtype='float32')
    # every anchor will get a set of parameterized bb vectors
    # but they are only meaningful to the positive boxes
    # only the regression loss from the positive boxes will be considered
    featuremap_boxes[inside_ind, :] = anchor_gt_boxes
    featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, cfg.RPN.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes

def get_train_dataflow():
    """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)
    """

    prw = PRWDataset(cfg.DATA.BASEDIR)
    imgs = prw.load()
    """
    To train on your own data, change this to your loader.
    Produce "imgs" as a list of dict, in the dict the following keys are needed for training:
    height, width: integer
    file_name: str, full path to the image
    boxes: numpy array of kx4 floats
    class: numpy array of k integers
    is_crowd: k booleans. Use k False if you don't know what it means.
    """

    ds = DataFromList(imgs, shuffle=cfg.DATA.TEST.SHUFFLE)

    # imgaug.Flip(horiz=True)
    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])

    def preprocess(img):
        fname, boxes, klass, is_crowd, re_id_class = img['file_name'], img['boxes'], \
                                                     img['class'], img['is_crowd'], img['re_id_class']
        boxes = np.copy(boxes)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        orig_shape = im.shape[:2]
        orig_im = np.copy(im)
        assert im is not None, fname
        im = im.astype('float32')
        # assume floatbox as input
        assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

        # augmentation:
        im, params = aug.augment_return_params(im)
        points = box_to_point8(boxes)
        points = aug.augment_coords(points, params)
        boxes = point8_to_box(points)
        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"

        # rpn anchor:
        try:
            # anchor_labels, anchor_boxes
            anchor_inputs = get_rpn_anchor_input(im, boxes, is_crowd)
            assert len(anchor_inputs) == 2

            boxes = boxes[is_crowd == 0]    # skip crowd boxes in training target
            klass = klass[is_crowd == 0]
            if not len(boxes):
                raise MalformedData("No valid gt_boxes!")
        except MalformedData as e:
            log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
            return None

        gt_id_prob_dist = np.zeros([len(re_id_class), cfg.DATA.NUM_ID])
        for obj_index, identity in enumerate(re_id_class):
            if identity != -2:
                gt_id_prob_dist[obj_index, identity - 1] = 1.
        gt_id_prob_dist[re_id_class == -2] = np.ones(cfg.DATA.NUM_ID, dtype=float) / cfg.DATA.NUM_ID

        ret = [im] + list(anchor_inputs) + [boxes, klass, re_id_class, gt_id_prob_dist, orig_shape, orig_im]

        return ret

    ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    return ds 

def get_eval_dataflow(shard=0, num_shards=1):
    """
    Args:
        shard, num_shards: to get subset of evaluation data
    """
    prw = PRWDataset(cfg.DATA.BASEDIR)
    if cfg.RE_ID.USE_DPM:
        imgs = prw.load_dpm()
        ds = DataFromListOfDict(imgs, ['file_name', 'file_name', 'boxes'])
    else:
        imgs = prw.load('test')
        ds = DataFromListOfDict(imgs, ['file_name', 'file_name', 'file_name'])
    num_imgs = len(imgs)

    # no filter for training
    # test if it can repeat keys
    
    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    # Evaluation itself may be multi-threaded, therefore don't add prefetch here.
    return ds

def get_query_dataflow():
    """
    Args:
        shard, num_shards: to get subset of evaluation data
    """
    prw = PRWDataset(cfg.DATA.BASEDIR)
    imgs = prw.load_query()

    # no filter for training
    # test if it can repeat keys
    ds = DataFromList(imgs, shuffle=False)

    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])

    def preprocess(img):
        fname, boxes, re_id_class = img['file_name'], img['boxes'], img['re_id_class']
        orig_boxes = np.copy(boxes)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = im.astype('float32')
        # assume floatbox as input
        assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

        # augmentation:
        im, params = aug.augment_return_params(im)
        points = box_to_point8(boxes)
        points = aug.augment_coords(points, params)
        boxes = point8_to_box(points)
        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"

        ret = [fname, im, boxes, re_id_class, orig_boxes]

        return ret

    ds = MapData(ds, preprocess)
    return ds

def get_train_aseval_dataflow():
    """
    Args:
        shard, num_shards: to get subset of evaluation data
    """
    prw = PRWDataset(cfg.DATA.BASEDIR)
    imgs = prw.load()

    # no filter for training
    # test if it can repeat keys
    ds = DataFromList(imgs, shuffle=False)

    aug = imgaug.AugmentorList(
        [CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)])

    def preprocess(img):
        fname = img['file_name']
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        orig_shape = im.shape[:2]
        assert im is not None, fname
        im = im.astype('float32')

        # augmentation:
        im, params = aug.augment_return_params(im)

        ret = [fname, im, orig_shape]

        return ret

    ds = MapData(ds, preprocess)
    return ds 