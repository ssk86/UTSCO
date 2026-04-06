# Copyright (c) Facebook, Inc. and its affiliates.
from .coco_evaluation import COCOEvaluator
from .visal_eval import PascalVOCDetectionEvaluator

# __all__ = [k for k in globals().keys() if not k.startswith("_")]

__all__ = [
    "COCOEvaluator",
    "PascalVOCDetectionEvaluator"
]
