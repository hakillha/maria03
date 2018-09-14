import os
import pprint

from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret) # ???
        return ret

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self):
        self._freezed = True

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------
_C.MODE_MASK = False        # FasterRCNN or MaskRCNN
_C.MODE_FPN = False

# dataset -----------------------
_C.DATA.BASEDIR = '../PRW-v16.04.20'
_C.DATA.TRAIN = ['train2014', 'valminusminival2014']   # i.e., trainval35k
_C.DATA.VAL = 'minival2014'   # For now, only support evaluation on single dataset
_C.DATA.NUM_CATEGORY = 1    # 80 categories. Plus bg or not?
_C.DATA.CLASS_NAMES = []  # NUM_CLASS (NUM_CATEGORY+1) strings, to be populated later by data loader. The first is BG.
_C.DATA.TEST.SHUFFLE = True

# basemodel ----------------------
_C.BACKBONE.WEIGHTS = 'ckpt/ImageNet-R50-AlignPadding.npz'   # /path/to/weights.npz
_C.BACKBONE.RESNET_NUM_BLOCK = [3, 4, 6, 3]     # for resnet50
# RESNET_NUM_BLOCK = [3, 4, 23, 3]    # for resnet101
_C.BACKBONE.FREEZE_AFFINE = False   # do not train affine parameters inside norm layers
_C.BACKBONE.NORM = 'FreezeBN'  # options: FreezeBN, SyncBN, GN
_C.BACKBONE.FREEZE_AT = 2  # options: 0, 1, 2

# Use a base model with TF-preferred padding mode,
# which may pad more pixels on right/bottom than top/left.
# See https://github.com/tensorflow/tensorflow/issues/18213

# In tensorpack model zoo, ResNet models with TF_PAD_MODE=False are marked with "-AlignPadding".
# All other models under `ResNet/` in the model zoo are trained with TF_PAD_MODE=True.
_C.BACKBONE.TF_PAD_MODE = False
_C.BACKBONE.STRIDE_1X1 = False  # True for MSRA models

# schedule -----------------------
_C.TRAIN.NUM_GPUS = None         # by default, will be set from code
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1e-2  # defined for a total batch size of 8. Otherwise it will be adjusted automatically
_C.TRAIN.WARMUP = 1000   # in terms of iterations. This is not affected by #GPUs
_C.TRAIN.STEPS_PER_EPOCH = 500

# Schedule means "steps" only when total batch size is 8.
# Otherwise the actual steps to decrease learning rate are computed from the schedule.
# LR_SCHEDULE = [120000, 160000, 180000]  # "1x" schedule in detectron
_C.TRAIN.LR_SCHEDULE = [240000, 320000, 360000]    # "2x" schedule in detectron
_C.TRAIN.NUM_EVALS = 20  # number of evaluations to run during training

# preprocessing --------------------
# Alternative old (worse & faster) setting: 600, 1024
# _C.PREPROC.SHORT_EDGE_SIZE = 800
# _C.PREPROC.MAX_SIZE = 1333
_C.PREPROC.SHORT_EDGE_SIZE = 600
_C.PREPROC.MAX_SIZE = 1024
# mean and std in RGB order.
# Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
_C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

# anchors -------------------------
_C.RPN.ANCHOR_STRIDE = 16 # as a relative base size?
_C.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)   # sqrtarea of the anchor box
_C.RPN.ANCHOR_RATIOS = (0.5, 1., 2.)
_C.RPN.POSITIVE_ANCHOR_THRESH = 0.7
_C.RPN.NEGATIVE_ANCHOR_THRESH = 0.3 # no hard negatives?

# rpn training -------------------------
_C.RPN.FG_RATIO = 0.5  # fg ratio among selected RPN anchors
_C.RPN.BATCH_PER_IM = 256  # total (across FPN levels) number of anchors that are marked valid
_C.RPN.MIN_SIZE = 0 # if either side of a proposal is smaller than this then it's ruled out
_C.RPN.PROPOSAL_NMS_THRESH = 0.7
_C.RPN.CROWD_OVERLAP_THRESH = 0.7  # boxes overlapping crowd will be ignored.
_C.RPN.HEAD_DIM = 1024      # used in C4 only; look into faster rcnn paper

# RPN proposal selection -------------------------------
# for C4
_C.RPN.TRAIN_PRE_NMS_TOPK = 12000
_C.RPN.TRAIN_POST_NMS_TOPK = 2000
_C.RPN.TEST_PRE_NMS_TOPK = 6000
_C.RPN.TEST_POST_NMS_TOPK = 1000   # if you encounter OOM in inference, set this to a smaller number

# fastrcnn training ---------------------
_C.FRCNN.BATCH_PER_IM = 256
_C.FRCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]  # Better but non-standard setting: [20, 20, 10, 10]
_C.FRCNN.FG_THRESH = 0.5
_C.FRCNN.FG_RATIO = 0.25  # fg ratio in a ROI batch

# re-id branch
# _C.DATA.NUM_ID = 934
# unlabeled pedes shouldn't be counted
_C.DATA.NUM_ID = 933
_C.DATA.INCLUDE_ALL = True

_C.RE_ID.IOU_THRESH = 0.7
_C.RE_ID.NMS = True
_C.RE_ID.QUERY_EVAL = False
# _C.RE_ID.LOSS_NORMALIZATION = 1.0
_C.RE_ID.LOSS_NORMALIZATION = 4.5
# a small constant loss for re-id head when there is no det to stablize the moving average?
# or should it be a large value instead?
# _C.RE_ID.STABLE_LOSS = 1e-3
# _C.RE_ID.STABLE_LOSS = 10.0
_C.RE_ID.STABLE_LOSS = 0.1
_C.RE_ID.FC_LAYERS_ON = True
_C.RE_ID.COSINE_SOFTMAX = False
_C.RE_ID.LSRO = False

# testing -----------------------
_C.TEST.FRCNN_NMS_THRESH = 0.5

# Smaller threshold value gives significantly better mAP. But we use 0.05 for consistency with Detectron.
# mAP with 1e-4 threshold can be found at https://github.com/tensorpack/tensorpack/commit/26321ae58120af2568bdbf2269f32aa708d425a8#diff-61085c48abee915b584027e1085e1043  # noqa
_C.TEST.RESULT_SCORE_THRESH = 0.05
_C.TEST.RESULT_SCORE_THRESH_VIS = 0.3   # only visualize confident results
_C.TEST.RESULTS_PER_IM = 100

def finalize_configs(is_training):
    """
    Run some sanity checks, and populate some configs from others
    """
    _C.DATA.NUM_CLASS = _C.DATA.NUM_CATEGORY + 1  # +1 background
    _C.DATA.BASEDIR = os.path.expanduser(_C.DATA.BASEDIR)

    assert _C.BACKBONE.NORM in ['FreezeBN', 'SyncBN', 'GN'], _C.BACKBONE.NORM
    if _C.BACKBONE.NORM != 'FreezeBN':
        assert not _C.BACKBONE.FREEZE_AFFINE
    assert _C.BACKBONE.FREEZE_AT in [0, 1, 2]

    _C.RPN.NUM_ANCHOR = len(_C.RPN.ANCHOR_SIZES) * len(_C.RPN.ANCHOR_RATIOS)

    if is_training:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = '1'

        # setup NUM_GPUS
        assert 'OMPI_COMM_WORLD_SIZE' not in os.environ
        ngpu = get_num_gpu()
        assert ngpu % 8 == 0 or 8 % ngpu == 0, ngpu
        if _C.TRAIN.NUM_GPUS is None:
            _C.TRAIN.NUM_GPUS = ngpu
        else:
            assert _C.TRAIN.NUM_GPUS <= ngpu
    else:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

    _C.freeze()
    logger.info("Config: ------------------------------------------\n" + str(_C))