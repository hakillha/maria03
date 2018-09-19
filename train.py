import argparse
import cv2
import itertools
import json
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import tqdm
from concurrent.futures import ThreadPoolExecutor

from tensorpack import *
from tensorpack.models import FullyConnected
from tensorpack.tfutils.summary import add_moving_summary, add_tensor_summary
import tensorpack.utils.viz as tpviz

from basemodel import (image_preprocess, resnet_c4_backbone,
                        resnet_conv5,)
from config import finalize_configs, config as cfg
# from data import PRWDataset
from data import (get_train_dataflow, get_all_anchors,
                  get_eval_dataflow, get_query_dataflow,
                  get_train_aseval_dataflow)
from eval import (detect_one_image, eval_output,
                  DetectionResult, query_eval_output,
                  classifier_eval_output,
                  # EvalCallback
                  )
from model_box import (RPNAnchors, roi_align,
                       encode_bbox_target, crop_and_resize,
                       decode_bbox_target,clip_boxes)
from model_frcnn import (sample_fast_rcnn_targets, fastrcnn_outputs,
                         fastrcnn_losses, fastrcnn_predictions,
                         fastrcnn_predictions_id)
from model_id import query_eval
from model_rpn import (rpn_head, generate_rpn_proposals,
                        rpn_losses,)
from viz import draw_final_outputs
from utils.box_ops import tf_clip_boxes, pairwise_iou


class DetectionModel(ModelDesc):

    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def fastrcnn_training(self, image,
                          rcnn_labels, fg_rcnn_boxes, gt_boxes_per_fg,
                          rcnn_label_logits, fg_rcnn_box_logits):
        """
        Args:
            image (NCHW):
            rcnn_labels (n): labels for each sampled targets
            fg_rcnn_boxes (fg x 4): proposal boxes for each sampled foreground targets
            gt_boxes_per_fg (fg x 4): matching gt boxes for each sampled foreground targets
            rcnn_label_logits (n): label logits for each sampled targets
            fg_rcnn_box_logits (fg x #class x 4): box logits for each sampled foreground targets
        """

        with tf.name_scope('fg_sample_patch_viz'):
            fg_sampled_patches = crop_and_resize(
                image, fg_rcnn_boxes,
                tf.zeros([tf.shape(fg_rcnn_boxes)[0]], dtype=tf.int32), 300)
            fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
            fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
            tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

        encoded_boxes = encode_bbox_target(
            gt_boxes_per_fg, fg_rcnn_boxes) * tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32)
        fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
            rcnn_labels, rcnn_label_logits,
            encoded_boxes,
            fg_rcnn_box_logits)
        return fastrcnn_label_loss, fastrcnn_box_loss

    def fastrcnn_inference(self, image_shape2d,
                           rcnn_boxes, rcnn_label_logits, rcnn_box_logits):
        """
        Args:
            image_shape2d: h, w
            rcnn_boxes (nx4): the proposal boxes
            rcnn_label_logits (n):
            rcnn_box_logits (nx #class x 4):

        Returns:
            boxes (mx4):
            labels (m): each >= 1
        """
        rcnn_box_logits = rcnn_box_logits[:, 1:, :] # throw away the bg logit
        # we can see the bg is not included as a class here
        # print(rcnn_box_logits.shape)
        rcnn_box_logits.set_shape([None, cfg.DATA.NUM_CATEGORY, None])
        # print(rcnn_label_logits.shape)
        # tf.nn.softmax has a default -1 (last) axis
        label_probs = tf.nn.softmax(rcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
        anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, cfg.DATA.NUM_CATEGORY, 1])   # #proposal x #Cat x 4
        decoded_boxes = decode_bbox_target(
            rcnn_box_logits /
            tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32), anchors)
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

        # indices: Nx2. Each index into (#proposal, #category)
        pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
        final_probs = tf.identity(final_probs, 'final_probs')
        final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
        final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
        return final_boxes, final_labels, final_probs

    def fastrcnn_inference_id(self, image_shape2d,
                              rcnn_boxes, rcnn_label_logits, rcnn_box_logits):
        """
        Args:
            image_shape2d: h, w
            rcnn_boxes (nx4): the proposal boxes
            rcnn_label_logits (n):
            rcnn_box_logits (nx #class x 4):

        Returns:
            boxes (mx4):
            labels (m): each >= 1
        """
        rcnn_box_logits = rcnn_box_logits[:, 1:, :] # throw away the bg logit
        # we can see the bg is not included as a class here
        # print(rcnn_box_logits.shape)
        rcnn_box_logits.set_shape([None, cfg.DATA.NUM_CATEGORY, None])
        # print(rcnn_label_logits.shape)
        # tf.nn.softmax has a default -1 (last) axis
        label_probs = tf.nn.softmax(rcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
        anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, cfg.DATA.NUM_CATEGORY, 1])   # #proposal x #Cat x 4
        decoded_boxes = decode_bbox_target(
            rcnn_box_logits /
            tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32), anchors)
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

        # indices: Nx2. Each index into (#proposal, #category)
        pred_indices, final_probs = fastrcnn_predictions_id(decoded_boxes, label_probs)
        final_probs = tf.identity(final_probs, 'final_probs')
        final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
        final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
        return final_boxes, final_labels, final_probs

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['final_boxes', 'final_probs', 'final_labels', 'feature_vector']
        return ['image'], out

    def get_query_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        return ['image', 'gt_boxes'], ['feature_vector']

    def get_classifier_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['rescaled_final_boxes', 're_id_probs']
        # out = ['rescaled_final_boxes', 're_id_probs', 're_boxes_pre_clip']
        return ['image', 'orig_shape'], out


class ResNetC4Model(DetectionModel):
    
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels'),  # all > 0
            tf.placeholder(tf.int64, (None,), 'gt_ids'),
            tf.placeholder(tf.int64, (None, cfg.DATA.NUM_ID), 'gt_id_prob_dist'),
            tf.placeholder(tf.int64, (2,), 'orig_shape')]
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_ids, gt_id_prob_dist, orig_shape = inputs
        image = self.preprocess(image)     # 1CHW

        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)

        anchors = RPNAnchors(get_all_anchors(), anchor_labels, anchor_boxes)
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]     # h,w
        # decode into actual image coordinates
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if is_training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if is_training else cfg.RPN.TEST_POST_NMS_TOPK)

        if is_training:
            # sample proposal boxes in training
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
        else:
            # The boxes to be used to crop RoIs.
            # Use all proposal boxes in inference
            rcnn_boxes = proposal_boxes

        boxes_on_featuremap = rcnn_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        # size? #proposals*h*w*c?
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
        
        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)

        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            # outputs from fg proposals
            fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
            fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

            # rcnn_labels: the labels of the proposals
            # fg_sampled_boxes: fg proposals
            # matched_gt_boxes: just like RPN, the gt boxes
            #                   that match the corresponding fg proposals
            fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_training(
                image, rcnn_labels, fg_sampled_boxes,
                matched_gt_boxes, fastrcnn_label_logits, fg_fastrcnn_box_logits)


            # acquire pred for re-id training
            # turning NMS off gives re-id branch more training samples
            if cfg.RE_ID.NMS:
                boxes, final_labels, final_probs = self.fastrcnn_inference(
                    image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)
            else:
                boxes, final_labels, final_probs = self.fastrcnn_inference_id(
                    image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)
            # scale = tf.sqrt(tf.cast(image_shape2d[0], tf.float32) / tf.cast(orig_shape[0], tf.float32) * 
            #                 tf.cast(image_shape2d[1], tf.float32) / tf.cast(orig_shape[1], tf.float32))
            # final_boxes = boxes / scale
            # # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
            # final_boxes = tf_clip_boxes(final_boxes, orig_shape)

            # IOU, discard bad dets, assign re-id labels
            # the results are already NMS so no need to NMS again
            # crop from conv4 with dets (maybe plus gts)
            # feedforward re-id branch
            # resizing during ROIalign?
            iou = pairwise_iou(boxes, gt_boxes) # are the gt boxes resized?
            tp_mask = tf.reduce_max(iou, axis=1) >= cfg.RE_ID.IOU_THRESH
            iou = tf.boolean_mask(iou, tp_mask)
            # return iou to debug

            def re_id_loss(pred_boxes, pred_matching_gt_ids,
                           pred_matching_gt_id_dist, featuremap):
                with tf.variable_scope('id_head'):
                    num_of_samples_used = tf.get_variable('num_of_samples_used', initializer=0, trainable=False)
                    num_of_samples_used = num_of_samples_used.assign_add(tf.shape(pred_boxes)[0])

                    boxes_on_featuremap = pred_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
                    # name scope?
                    # stop gradient
                    roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
                    feature_idhead = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
                    feature_gap = GlobalAvgPooling('gap', feature_idhead, data_format='channels_first')

                    if cfg.RE_ID.FC_LAYERS_ON:
                        init = tf.variance_scaling_initializer()
                        # first dimension of the output tensor being batch size
                        hidden = FullyConnected('fc6', feature_gap, 1024, kernel_initializer=init, activation=tf.nn.relu)
                        hidden = FullyConnected('fc7', hidden, 1024, kernel_initializer=init, activation=tf.nn.relu)
                        hidden = FullyConnected('fc8', hidden, 256, kernel_initializer=init, activation=tf.nn.relu)
                    else:
                        hidden = feature_gap

                    if cfg.RE_ID.COSINE_SOFTMAX:
                        mean_vectors = tf.get_variable('mean_vectors', (hidden.shape[-1], int(cfg.DATA.NUM_ID)),
                            initializer=tf.truncated_normal_initializer(stddev=1e-3), regularizer=None)
                        # log cos_scale
                        cos_scale = tf.get_variable('cos_scale', (), tf.float32,
                            initializer=tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(1e-1))
                        cos_scale = tf.nn.softplus(cos_scale)

                        mean_vectors = tf.nn.l2_normalize(mean_vectors, axis=0)
                        id_logits = cos_scale * tf.matmul(hidden, mean_vectors)
                    else:
                        id_logits = FullyConnected(
                            'class', hidden, cfg.DATA.NUM_ID,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01))

                # use sparse into dense to create 1 one vector
                if cfg.RE_ID.LSRO:
                    label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                labels=pred_matching_gt_id_dist, logits=id_logits)
                else:
                    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                labels=pred_matching_gt_ids, logits=id_logits)
                label_loss = tf.reduce_mean(label_loss, name='label_loss')

                return label_loss, num_of_samples_used

            def check_unid_pedes(iou, gt_ids, boxes, tp_mask,
                                 featuremap):
                pred_gt_ind = tf.argmax(iou, axis=1)
                # output following tensors
                # pick out the -2 class here
                pred_matching_gt_ids = tf.gather(gt_ids, pred_gt_ind)
                pred_matching_gt_id_dist = tf.gather(gt_id_prob_dist, pred_gt_ind)
                pred_boxes = tf.boolean_mask(boxes, tp_mask)
                # label 1 corresponds to unid pedes
                labeled_id_ind = tf.not_equal(pred_matching_gt_ids, -2)
                # if cfg.RE_ID.LSRO:
                #     cond on if both types are 
                #     labeled_pred_matching_gt_ids = tf.boolean_mask(pred_matching_gt_ids, labeled_id_ind)
                #     pred_matching_labeled_gt_dist = tf.to_float(tf.map_fn(lambda x: tf.sparse_to_dense(x - 1, [cfg.DATA.NUM_ID], 1), 
                #                                     labeled_pred_matching_gt_ids))
                #     unlabeled_id_ind = tf.logical_not(labeled_id_ind)
                #     pred_matching_unlabeled_gt_dist = tf.map_fn(lambda x: x / cfg.DATA.NUM_ID, 
                #         tf.ones((tf.shape(tf.where(unlabeled_id_ind))[0], cfg.DATA.NUM_ID), dtype=tf.float32))
                        
                #     # gt_prob_dist = tf.scatter_update(gt_prob_dist, tf.where(labeled_id_ind), pred_matching_labeled_gt_dist)
                #     # need to redefine gt_prob_dist as a var to make it work
                #     # if I can scatter_update then I don't have to rearrange the box tensor as well 
                #     # keep all confident pede boxes as opposed to no LSRO
                #     labeled_pred_boxes = tf.boolean_mask(pred_boxes, labeled_id_ind)
                #     unlabeled_pred_boxes = tf.boolean_mask(pred_boxes, unlabeled_id_ind)
                #     pred_matching_gt_dist = tf.reshape(tf.stack([pred_matching_labeled_gt_dist, 
                #                                                  pred_matching_unlabeled_gt_dist]), [-1, cfg.DATA.NUM_ID])
                #     pred_boxes = tf.reshape(tf.stack([labeled_pred_boxes,
                #                                       unlabeled_pred_boxes]), [-1, 4])
                #     pred_matching_gt_ids = None
                # else:
                if not cfg.RE_ID.LSRO: # then throw away unlabeled boxes and ids
                    pred_matching_gt_ids = tf.boolean_mask(pred_matching_gt_ids, labeled_id_ind) 
                    pred_boxes = tf.boolean_mask(pred_boxes, labeled_id_ind)
                    pred_matching_gt_dist = None

                ret = tf.cond(tf.equal(tf.size(pred_boxes), 0), 
                              lambda: (tf.constant(cfg.RE_ID.STABLE_LOSS), tf.constant(0)),
                              lambda: re_id_loss(pred_boxes, pred_matching_gt_ids,
                                                 pred_matching_gt_id_dist, featuremap))
                return ret

            with tf.name_scope('id_head'):
                # no detection has IOU > 0.7, re-id returns 0 loss
                re_id_loss, num_of_samples_used = tf.cond(tf.equal(tf.size(iou), 0), 
                    lambda: (tf.constant(cfg.RE_ID.STABLE_LOSS), tf.constant(0)),
                    lambda: check_unid_pedes(iou, gt_ids, 
                        boxes, tp_mask, featuremap))

                add_tensor_summary(num_of_samples_used, ['scalar'], name='num_of_samples_used')
            tf.add_to_collection('re_id_summaries_misc', num_of_samples_used)
            # for debug, use tensor name to take out the handle
            # return re_id_loss

            # pred_gt_ind = tf.argmax(iou, axis=1)
            # # output following tensors
            # # pick out the -2 class here
            # pred_gt_ids = tf.gather(gt_ids, pred_gt_ind)
            # pred_boxes = tf.boolean_mask(boxes, tp_mask)
            # unid_ind = pred_gt_ids != 1
            
            # return unid_ind

            # return tf.shape(boxes)[0]

            unnormed_id_loss = tf.identity(re_id_loss, name='unnormed_id_loss')
            re_id_loss = tf.divide(re_id_loss, cfg.RE_ID.LOSS_NORMALIZATION, 're_id_loss')
            add_moving_summary(unnormed_id_loss)
            # add_moving_summary(re_id_loss)

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

            # weights on the losses?
            total_cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                re_id_loss,
                wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            if cfg.RE_ID.QUERY_EVAL:
                # resize the gt_boxes in dataflow
                final_boxes = gt_boxes
            else:
                final_boxes, final_labels, _ = self.fastrcnn_inference(
                    image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)

            with tf.variable_scope('id_head'):
                preds_on_featuremap = final_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
                # name scope?
                # stop gradient
                roi_resized = roi_align(featuremap, preds_on_featuremap, 14)
                feature_idhead = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
                feature_gap = GlobalAvgPooling('gap', feature_idhead, data_format='channels_first')

                if cfg.RE_ID.FC_LAYERS_ON:
                    hidden = FullyConnected('fc6', feature_gap, 1024, activation=tf.nn.relu)
                    hidden = FullyConnected('fc7', hidden, 1024, activation=tf.nn.relu)
                    fv = FullyConnected('fc8', hidden, 256, activation=tf.nn.relu)
                else:
                    fv = feature_gap

                if cfg.RE_ID.COSINE_SOFTMAX:
                    # do we need to consider reuse here? 
                    # no cuz these are 2 different branches of the if control flow statement, 
                    # i.e. we are not trying to use the same var by repeatedly defining/annoucing them?
                    mean_vectors = tf.get_variable('mean_vectors', (fv.shape[-1], int(cfg.DATA.NUM_ID)),
                        initializer=tf.truncated_normal_initializer(stddev=1e-3), regularizer=None)
                    # log cos_scale
                    cos_scale = tf.get_variable('cos_scale', (), tf.float32,
                        initializer=tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(1e-1))
                    cos_scale = tf.nn.softplus(cos_scale)

                    mean_vectors = tf.nn.l2_normalize(mean_vectors, axis=0)
                    id_logits = cos_scale * tf.matmul(fv, mean_vectors)
                else:
                    id_logits = FullyConnected(
                        'class', fv, cfg.DATA.NUM_ID,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

            scale = tf.sqrt(tf.cast(image_shape2d[0], tf.float32) / tf.cast(orig_shape[0], tf.float32) * 
                            tf.cast(image_shape2d[1], tf.float32) / tf.cast(orig_shape[1], tf.float32))
            rescaled_final_boxes = final_boxes / scale
            # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
            # rescaled_final_boxes_pre_clip = tf.identity(rescaled_final_boxes, name='re_boxes_pre_clip')
            rescaled_final_boxes = tf_clip_boxes(rescaled_final_boxes, orig_shape)
            rescaled_final_boxes = tf.identity(rescaled_final_boxes, 'rescaled_final_boxes')

            fv = tf.identity(fv, name='feature_vector')
            prob = tf.nn.softmax(id_logits, name='re_id_probs')

def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_output(
        df, lambda img: detect_one_image(img, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    # print_evaluation_scores(output_file)

def query_evaluate(pred_func, output_file):
    df = get_query_dataflow()
    all_results = query_eval_output(df, pred_func)
    with open(output_file, 'w') as f:
        json.dump(all_results, f)

def classifier_evaluate(pred_func, output_file):
    df = get_train_aseval_dataflow()
    all_results = classifier_eval_output(df, pred_func)
    with open(output_file, 'w') as f:
        json.dump(all_results, f)

def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    # tpviz.interactive_imshow(viz)
    cv2.imwrite(os.path.basename(input_file), viz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', default='train_log/fasterrcnn')
    parser.add_argument('--evaluate', help='Run evaluation on PRW. '
                                           'This argument is the path to the output json results file.')
    parser.add_argument('--evaluate_classifier')
    parser.add_argument('--predict', help='Single image inference. This argument points to a image file or folder.')
    parser.add_argument('--query', help='Path to the output json file.')
    # parser.add_argument('--re-id_training', action='store_true', default=True)
    parser.add_argument('--random_predict', action='store_true')
    parser.add_argument('--config', nargs='+')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--additional_monitoring', action='store_true')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()

    if args.evaluate or args.predict:
        assert args.modeldir
        finalize_configs(is_training=False)

        if args.predict:
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        pred = OfflinePredictor(PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.modeldir),
                input_names=MODEL.get_inference_tensor_names()[0],
                output_names=MODEL.get_inference_tensor_names()[1]))
        if args.evaluate:
            assert args.evaluate.endswith('.json'), args.evaluate
            offline_evaluate(pred, args.evaluate)
        elif args.predict:
            cfg.DATA.CLASS_NAMES = ['BG', 'pedestrian']
            if args.random_predict:
                predict_file_list = os.listdir(args.predict)
                predict_file = os.path.join(args.predict, predict_file_list[random.randint(0, len(predict_file_list))])
                print(predict_file)
                predict(pred, predict_file)
            else:
                predict(pred, args.predict)
    elif args.evaluate_classifier:
        assert args.modeldir
        finalize_configs(is_training=False)
        pred = OfflinePredictor(PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.modeldir),
                input_names=MODEL.get_classifier_tensor_names()[0],
                output_names=MODEL.get_classifier_tensor_names()[1]))
        assert args.evaluate_classifier.endswith('.json'), args.evaluate_classifier
        classifier_evaluate(pred, args.evaluate_classifier)
    elif args.query:
        assert args.modeldir
        cfg.RE_ID.QUERY_EVAL = True
        finalize_configs(is_training=False)
        pred = OfflinePredictor(PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.modeldir),
                input_names=MODEL.get_query_inference_tensor_names()[0],
                output_names=MODEL.get_query_inference_tensor_names()[1]))
        assert args.query.endswith('.json'), args.query
        query_evaluate(pred, args.query)
    else:
        logger.set_logger_dir(args.logdir, 'b')

        finalize_configs(is_training=True)
        stepnum = cfg.TRAIN.STEPS_PER_EPOCH

        # warmup is step based, lr is epoch based
        # why lower lr if gradients are averaged over #gpus
        # also is it a factor that each gpu has a smaller batch relatively
        # warmup lr?
        init_lr = cfg.TRAIN.BASE_LR * 0.33 * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
        warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
        warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
        lr_schedule = [(int(np.ceil(warmup_end_epoch)), cfg.TRAIN.BASE_LR)]

        factor = 8. / cfg.TRAIN.NUM_GPUS
        for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
            mult = 0.1 ** (idx + 1)
            lr_schedule.append(
                (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
        logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
        logger.info("LR Schedule (epochs, value): " + str(lr_schedule))

        callbacks = [
            PeriodicCallback(
                ModelSaver(max_to_keep=2, keep_checkpoint_every_n_hours=6),
                every_k_epochs=20),
            # linear warmup
            ScheduledHyperParamSetter(
                'learning_rate', warmup_schedule, interp='linear', step_based=True),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            # EvalCallback(*MODEL.get_inference_tensor_names()),
            PeakMemoryTracker(),
            EstimatedTimeLeft(median=True),
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
            GPUUtilizationTracker(),
            # MergeAllSummaries(period=1, key='re_id_summaries_misc')
        ]

        if args.modeldir:
            session_init = get_model_loader(args.modeldir)
        else:
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

        if args.additional_monitoring:
            extra_callbacks = [MergeAllSummaries(period=1)]
            extra_monitors = [ScalarPrinter(enable_step=True, whitelist=['num_of_samples_used', 'loss'])]
        else:
            extra_callbacks = []
            extra_monitors = []

        debug_mode = args.debug_mode
        if not debug_mode:
            traincfg = TrainConfig(
                model=MODEL,
                data=QueueInput(get_train_dataflow()),
                callbacks=callbacks + extra_callbacks,
                steps_per_epoch=stepnum,
                max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
                session_init=session_init,
                monitors=DEFAULT_MONITORS() + extra_monitors
            )
            # nccl mode has better speed than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
            launch_train_with_config(traincfg, trainer)
        else:
            # test01
            # df = get_train_dataflow()
            # df.reset_state()
            # df_gen = df.get_data()
            # with TowerContext('', is_training=True):
            #     model = ResNetC4Model()
            #     input_handle = model.inputs()
            #     ret_handle = model.build_graph(*input_handle)

            # with tf.Session() as sess:
            #     # print('Number of trainable parameters: \n')
            #     # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


            #     sess.run(tf.global_variables_initializer())
            #     session_init._setup_graph()
            #     session_init._run_init(sess)

            #     for _ in range(3):
            #         image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_ids, orig_shape, orig_im = next(df_gen)
            #         image_ = image.astype(np.int16)
            #         print(image)
            #         # print(orig_im)
            #         input_dict = {input_handle[0]: image,
            #                       input_handle[1]: anchor_labels,
            #                       input_handle[2]: anchor_boxes,
            #                       input_handle[3]: gt_boxes,
            #                       input_handle[4]: gt_labels,
            #                       input_handle[5]: gt_ids,
            #                       input_handle[6]: orig_shape}
            #         ret = sess.run(ret_handle, input_dict)
            #         # print(ret)
            #         _, boxes, probs, labels = ret
            #         results = [DetectionResult(*args) for args in zip(boxes, probs, labels)]
            #         # final = draw_final_outputs(orig_im, results)
            #         # viz = np.concatenate((orig_im, final), axis=1)
            #         # tpviz.interactive_imshow(viz)

            #         final = draw_final_outputs(image_, results)
            #         viz = np.concatenate((image_, final), axis=1)
            #         tpviz.interactive_imshow(viz)

                    # print(ret)
                    # for i in ret[1]:
                    #     print(i)

            # test02
            # df = get_eval_dataflow()
            # df.reset_state()
            # df_gen = df.get_data()
            # for _ in range(10):
            #     print(next(df_gen))

            # basic test
            df = get_train_dataflow()
            df.reset_state()
            df_gen = df.get_data()
            with TowerContext('', is_training=True):
                model = ResNetC4Model()
                input_handle = model.inputs()
                ret_handle = model.build_graph(*input_handle)

            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)
            for var in tf.trainable_variables():
                print(var.name)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                session_init._setup_graph()
                session_init._run_init(sess)
                for _ in range(10):
                    image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_ids, gt_id_prob_dist, orig_shape, orig_im = next(df_gen)
                    input_dict = {input_handle[0]: image,
                                  input_handle[1]: anchor_labels,
                                  input_handle[2]: anchor_boxes,
                                  input_handle[3]: gt_boxes,
                                  input_handle[4]: gt_labels,
                                  input_handle[5]: gt_ids,
                                  input_handle[6]: gt_id_prob_dist,
                                  input_handle[7]: orig_shape}
                    ret = sess.run(ret_handle, input_dict)
                    print(ret)

