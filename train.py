import argparse
import cv2
import numpy as np
import os
import os.path
import random
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
import tensorpack.utils.viz as tpviz

from basemodel import (image_preprocess, resnet_c4_backbone,
                        resnet_conv5,)
from config import finalize_configs, config as cfg
# from data import PRWDataset
from data import (get_train_dataflow, get_all_anchors,)
from eval import (detect_one_image)
from model_box import (RPNAnchors, roi_align,
                       encode_bbox_target, crop_and_resize,
                       decode_bbox_target,clip_boxes)
from model_frcnn import (sample_fast_rcnn_targets, fastrcnn_outputs,
                         fastrcnn_losses, fastrcnn_predictions)
from model_rpn import (rpn_head, generate_rpn_proposals,
                        rpn_losses,)
from viz import draw_final_outputs


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
        return final_boxes, final_labels

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['final_boxes', 'final_probs', 'final_labels']
        return ['image'], out


class ResNetC4Model(DetectionModel):
    
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')]  # all > 0
        return ret

    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
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

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

            # weights on the losses?
            total_cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_boxes, final_labels = self.fastrcnn_inference(
                image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)


def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_coco(
        df, lambda img: detect_one_image(img, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    print_evaluation_scores(output_file)

def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    # tpviz.interactive_imshow(viz)
    cv2.imwrite('test01.png', viz)


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
        self.dataflows = [get_eval_dataflow(shard=k, num_shards=self.num_predictor)
                          for k in range(self.num_predictor)]

    def _before_train(self):
        num_eval = cfg.TRAIN.NUM_EVALS
        interval = max(self.trainer.max_epoch // (num_eval + 1), 1)
        self.epochs_to_eval = set([interval * k for k in range(1, num_eval + 1)])
        self.epochs_to_eval.add(self.trainer.max_epoch)
        if len(self.epochs_to_eval) < 15:
            logger.info("[EvalCallback] Will evaluate at epoch " + str(sorted(self.epochs_to_eval)))
        else:
            logger.info("[EvalCallback] Will evaluate every {} epochs".format(interval))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', default='train_log/fasterrcnn')
    parser.add_argument('--evaluate', help='Run evaluation on PRW. '
                                           'This argument is the path to the output json results file')
    parser.add_argument('--predict', help='Single image inference. This argument points to a image file or folder.')
    parser.add_argument('--random_predict', action='store_true')
    parser.add_argument('--config', nargs='+')

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
            if args.random_predict:
                predict_file_list = os.listdir(args.predict)
                predict_file = os.path.join(args.predict, predict_file_list[random.randint(0, len(predict_file_list))])
                print(predict_file)
                predict(pred, predict_file)
            else:
                predict(pred, args.predict)
    else:
        logger.set_logger_dir(args.logdir, 'b')

        finalize_configs(is_training=True)
        stepnum = cfg.TRAIN.STEPS_PER_EPOCH

        # warmup is step based, lr is epoch based
        # why lower lr if gradients are averaged over #gpus
        # also is it a factor that each gpu has a smaller batch relatively
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
                ModelSaver(max_to_keep=5, keep_checkpoint_every_n_hours=2),
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
        ]

        if args.modeldir:
            session_init = get_model_loader(args.modeldir)
        else:
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

        dev_test = False
        if dev_test:
            df = get_train_dataflow()
            df.reset_state()
            df_gen = df.get_data()
            with TowerContext('', is_training=True):
                model = ResNetC4Model()
                input_handle = model.inputs()
                ret_handle = model.build_graph(*input_handle)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                session_init._run_init(sess)

                for _ in range(1000):
                    image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = next(df_gen)
                    input_dict = {input_handle[0]: image,
                                    input_handle[1]: anchor_labels,
                                    input_handle[2]: anchor_boxes,
                                    input_handle[3]: gt_boxes,
                                    input_handle[4]: gt_labels,}
                    ret = sess.run(ret_handle, input_dict)
                    print(ret)
                    # for i in ret[1]:
                    #     print(i)
        else:
            traincfg = TrainConfig(
                model=MODEL,
                data=QueueInput(get_train_dataflow()),
                callbacks=callbacks,
                steps_per_epoch=stepnum,
                max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
                session_init=session_init,
            )
            # nccl mode has better speed than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
            launch_train_with_config(traincfg, trainer)
