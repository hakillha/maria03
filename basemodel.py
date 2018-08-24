import numpy as np
import tensorflow as tf

from config import config as cfg

def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.PREPROC.PIXEL_MEAN
        std = np.asarray(cfg.PREPROC.PIXEL_STD)
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd
        return image