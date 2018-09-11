import tensorflow as tf

from tensorpack import *

from config import config as cfg
from basemodel import (resnet_c4_backbone, resnet_conv5)

def query_eval(img):
    featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
    with tf.variable_scope('id_head'):
        # the re-id conv5
        featuremap = resnet_conv5(featuremap, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
        feature_gap = GlobalAvgPooling('gap', featuremap, data_format='channels_first')

        hidden = FullyConnected('fc6', feature_gap, 1024, activation=tf.nn.relu)
        hidden = FullyConnected('fc7', hidden, 1024, activation=tf.nn.relu)
        fv = FullyConnected('fc8', hidden, 256, activation=tf.nn.relu)

    fv = tf.identity(fv, name='feature_vector')