import argparse
import tensorflow as tf

from tensorpack import *

from config import finalize_configs, config as cfg

class DetectionModel(ModelDesc):
    pass

class ResNetC4Model(DetectionModel):
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
            EvalCallback(*MODEL.get_inference_tensor_names()),
            PeakMemoryTracker(),
            EstimatedTimeLeft(median=True),
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
            GPUUtilizationTracker(),
        ]