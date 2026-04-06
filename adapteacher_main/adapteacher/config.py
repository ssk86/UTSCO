# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ateacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 2
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 2
    _C.SOLVER.FACTOR_LIST = (1,)
    
    _C.DATASETS.TRAIN_LABEL = ("source_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("target_train",)
    _C.DATASETS.CROSS_DATASET = True
    # _C.TEST.EVALUATOR = "voceval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.8
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 4
    _C.SEMISUPNET.BURN_UP_STEP = 8000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.9996
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
    _C.SEMISUPNET.INS_LOSS_WEIGHT = 0.3
    _C.SEMISUPNET.TAU_MIN = 0.0
    _C.SEMISUPNET.TAU_MAX = 0.5
    _C.SEMISUPNET.TAU_GAMMA = 2.0    

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 8  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/source_supervision.json"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
