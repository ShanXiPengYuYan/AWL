import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Path to dataset, could be overwritten by command line argument
_C.DATA.VOX1_DATA_PATH = 'your_dataset_path/vox1/wav'
_C.DATA.VOX2_DATA_PATH = 'your_dataset_path/vox2/wav'
_C.DATA.CN1_DATA_PATH = '/data/chenjunyu/data/CN/cn-celeb1'
_C.DATA.CN2_DATA_PATH = '/data/chenjunyu/data/CN/cn-celeb2'
_C.DATA.TEST_DATA_PATH = '/data/chenjunyu/data/CN/eval'
# Dataset list
_C.DATA.DATASET = ['vox1dev']
# Number of mel filterbanks
_C.DATA.N_MELS = 64
# Input length to the network for training
_C.DATA.MAX_FRAMES = 200
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
_C.DATA.MAX_SRG_PER_SPK = 800
_C.DATA.N_PER_SPEAKER = 1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type[ResNet, TDNN, Transformer]
_C.MODEL.TYPE = 'ResNet'
# Model name
_C.MODEL.NAME = 'ResNetSE34L'
# Aggregation mode
_C.MODEL.AGGREGATION = 'ASP'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.extract_embedding = None
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1211
# Embedding size in the last FC layer
_C.MODEL.EMB_SIZE = 512
_C.MODEL.DUL = False


# Loss function 
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.NAME = 'aamsoftmax'
_C.MODEL.LOSS.MARGIN = 0.2
_C.MODEL.LOSS.SCALE = 30
# ResNet parameters
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.DEPTHS = [3, 4, 6, 3]
_C.MODEL.RESNET.DIMS = [32, 64, 128, 256]
# configure of MHA
_C.MODEL.MHA = CN()
_C.MODEL.MHA.QK_SCALE = None
_C.MODEL.MHA.QKV_BIAS = True
_C.MODEL.MHA.DIM = 32

# residual fusion parameters
_C.MODEL.RF = CN()
_C.MODEL.RF.DEPTHS = [2, 2, 6, 2]
_C.MODEL.RF.DIMS = [32, 64, 128, 256]
_C.MODEL.RF.NUM_HEADS = [1, 2, 4, 8]
_C.MODEL.RF.QKV_BIAS = True
_C.MODEL.RF.QK_SCALE = None
_C.MODEL.RF.REDUCE_DIM = False
# cross 
_C.MODEL.CROSS = CN()
_C.MODEL.CROSS.GROUP_SIZE = 8
_C.MODEL.CROSS.DEPTHS = [2, 2, 6, 2]
_C.MODEL.CROSS.DIMS = [32, 64, 128, 256]
_C.MODEL.CROSS.NUM_HEADS = [1, 2, 4, 8]
_C.MODEL.CROSS.QKV_BIAS = True
_C.MODEL.CROSS.QK_SCALE = None
_C.MODEL.CROSS.REDUCE_DIM = True

# ECAPA
_C.MODEL.ECAPA = CN()
_C.MODEL.ECAPA.DIM = 1024
# CNN stem of ECAPA CNN-TDNN
_C.MODEL.CNN_STEM = CN()
_C.MODEL.CNN_STEM.DIM = 128

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# Batch size for GPU, could be overwritten by command line argument
_C.TRAIN.MUTI_INPUTS = True
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WEIGHT_DECAY = 2e-5

# warm up
_C.TRAIN.EPOCH_ITER = -1
_C.TRAIN.WARMUP_EPOCH = 1
# LR scheduler
_C.TRAIN.LR = 1e-3
_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.NAME = 'steplr'
# steplr
_C.TRAIN.SCHEDULER.STEPLR = CN()
_C.TRAIN.SCHEDULER.STEPLR.LR_STEP = 'epoch' 
_C.TRAIN.SCHEDULER.STEPLR.STEP_SIZE = 1
_C.TRAIN.SCHEDULER.STEPLR.DECAY_RATE = 0.97
# TODO multilr

# cosinelr
_C.TRAIN.SCHEDULER.COSINELR = CN()
_C.TRAIN.SCHEDULER.COSINELR.MIN_LR = 1e-8

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# SGD momentum
_C.TRAIN.OPTIMIZER.SGD_MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.FLAG = False
_C.AUG.MUSAN_PATH = 'your_dataset_path/musan_split'
_C.AUG.RIR_PATH = 'your_dataset_path/RIRS_NOISES/simulated_rirs'
_C.AUG.SPECAUG = CN()
_C.AUG.SPECAUG.FREQ_MASK_WIDTH = (0, 8)
_C.AUG.SPECAUG.TIME_MASK_WIDTH = (0, 10)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.FLAG = True
_C.TEST.DATASET = ['O']
_C.TEST.MODE = 'seg'
_C.TEST.SPEED_UP = True
# TODO
_C.TEST.BATCH_SIZE = 25
_C.TEST.SEG = CN()
_C.TEST.SEG.MAX_FRAMES = 400
_C.TEST.SEG.NUM_EVAL = 10
_C.TEST.AS_NORM = CN()
_C.TEST.AS_NORM.FLAG = False
_C.TEST.AS_NORM.TOPK = 1000
_C.TEST.MUTI = False

# _C.NISQA.mode = ''
# _C.NISQA.deg = ''
# _C.NISQA.data_dir = ''
# _C.NISQA.output_dir = ''
# _C.NISQA.csv_file = ''
# _C.NISQA.csv_deg = ''
# _C.NISQA.num_workers = ''
# _C.NISQA.bs = ''
# _C.NISQA.ms_channel = ''


# whether to save weights
_C.SAVE = True
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
# Fixed random seed
_C.SEED = 1314
# 
_C.GPU = '0,1'
_C.SAVE_FILE = 'log.txt'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.TRIAN.BATCH_SIZE = args.batch_size
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.augment:
        config.AUG.FLAG = args.augment
    if args.gpu:
        config.GPU = args.gpu
    # if args.model:
    #     config.MODEL.NAME = args.model

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
