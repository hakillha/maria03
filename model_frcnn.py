import tensorflow as tf

from tensorpack.models import (Conv2D, FullyConnected,
                                layer_register,)
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary

from config import config as cfg
from utils.box_ops import pairwise_iou

@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)

@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some ROIs from all proposals for training.
    #fg is guaranteed to be > 0, because ground truth boxes are added as RoIs.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t labels, in [0, #class-1]. Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
    	# (n+m)*1
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    # each entry is the index of the gt that this proposal overlaps the most
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)
    # stop the gradient -- they are meant to be training targets
    return tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'), \
        tf.stop_gradient(ret_labels, name='sampled_labels'), \
        tf.stop_gradient(fg_inds_wrt_gt)

@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class x 4)
    """
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    box_regression = FullyConnected(
        'box', feature, num_classes * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes, 4))
    return classification, box_regression

@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgxCx4
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds)
    indices = tf.stack(
        [tf.range(num_fg),
         tf.to_int32(fg_labels)], axis=1)  # #fgx2
    fg_box_logits = tf.gather_nd(fg_box_logits, indices)

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int32(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.truediv(num_zero, num_fg, name='false_negative')
        fg_accuracy = tf.reduce_mean(
            tf.gather(correct, fg_inds), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy, fg_accuracy, false_negative)
    return label_loss, box_loss