import argparse
import numpy as np
import tensorflow as tf

from tensorpack import *

from config import finalize_configs, config as cfg
# from data import PRWDataset
from data import get_train_dataflow


class DetectionModel(ModelDesc):

    def optimizer(self):
        # lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        # tf.summary.scalar('learning_rate-summary', lr)

        # # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        # lr = lr / 8.
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        # if cfg.TRAIN.NUM_GPUS < 8:
        #     opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        # return opt
        return tf.train.GradientDescentOptimizer(0.1)

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
        
        return inputs


class EvalCallback(Callback):
    """
    A callback that runs COCO evaluation once a while.
    It supports multi-GPU evaluation if TRAINER=='replicated' and single-GPU evaluation if TRAINER=='horovod'
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir')
    parser.add_argument('--logdir', default='train_log/fasterrcnn')
    parser.add_argument('--evaluate')
    parser.add_argument('--config', nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetC4Model()

    if args.evaluate:
        pass
    else:
        logger.set_logger_dir(args.logdir, 'd')

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


        # traincfg = TrainConfig(
        #     model=MODEL,
        #     data=QueueInput(get_train_dataflow()),
        #     callbacks=callbacks,
        #     steps_per_epoch=stepnum,
        #     max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        #     session_init=session_init,
        # )
        # # nccl mode has better speed than cpu mode
        # trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
        # launch_train_with_config(traincfg, trainer)

        dev_test = True
        if dev_test:
            df = get_train_dataflow()
            df.reset_state()
            df_gen = df.get_data()
            with TowerContext('', is_training=True):
                model = ResNetC4Model()
                input_handle = model.inputs()
                ret_handle = model.build_graph(input_handle)

            with tf.Session() as sess:
                for _ in range(10):
                    image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = next(df_gen)
                    input_dict = {input_handle[0]: image,
                                    input_handle[1]: anchor_labels,
                                    input_handle[2]: anchor_boxes,
                                    input_handle[3]: gt_boxes,
                                    input_handle[4]: gt_labels,}
                    ret = sess.run(ret_handle, input_dict)