# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

from detectron2.modeling.box_regression import Box2BoxTransform
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators
from detectron2.structures import pairwise_iou
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures import pairwise_iou
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog
import math
from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.engine.hooks import LossEvalHook, PeriodicCheckpointer
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation.visal_eval import PascalVOCDetectionEvaluator

from .probe import OpenMatchTrainerProbe
import copy
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T



def print_detectron2_params(model):
    print("\n" + "="*60)
    print("Parameter Breakdown (Student Model)")
    print("="*60)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    total = 0

    # backbone
    if hasattr(model, "backbone"):
        params = count(model.backbone)
        print(f"{'backbone':30s}: {params/1e6:.3f} M")
        total += params

    # RPN / proposal generator
    if hasattr(model, "proposal_generator"):
        params = count(model.proposal_generator)
        print(f"{'proposal_generator':30s}: {params/1e6:.3f} M")
        total += params

    # ROI Heads
    if hasattr(model, "roi_heads"):
        roi = model.roi_heads

        roi_total = count(roi)
        print(f"{'roi_heads (total)':30s}: {roi_total/1e6:.3f} M")

        # box head
        if hasattr(roi, "box_head"):
            params = count(roi.box_head)
            print(f"{'  ├─ box_head':30s}: {params/1e6:.3f} M")

        # box predictor
        if hasattr(roi, "box_predictor"):
            params = count(roi.box_predictor)
            print(f"{'  ├─ box_predictor':30s}: {params/1e6:.3f} M")

        # 你的 domain classifier（关键！！）
        for name, module in roi.named_children():
            if "domain" in name.lower() or "discriminator" in name.lower():
                params = count(module)
                print(f"{'  ├─ ' + name:30s}: {params/1e6:.3f} M")

        total += roi_total

    print("-"*60)
    print(f"{'Total (recomputed)':30s}: {total/1e6:.3f} M")

    real_total = count(model)
    print(f"{'Total (actual)':30s}: {real_total/1e6:.3f} M")
    print("="*60 + "\n")



@torch.no_grad()
def extract_gt_roi_features(model, batch):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    device = next(model.parameters()).device

    images = model.preprocess_image(batch)
    gt_instances = [x["instances"].to(device) for x in batch]

    features = model.backbone(images.tensor)
    gt_boxes = [inst.gt_boxes for inst in gt_instances]

    roi_features = model.roi_heads._shared_roi_transform(
        [features["res4"]], gt_boxes
    )
    box_features = model.roi_heads.box_head(roi_features)   # [N, D]
    box_features = F.normalize(box_features, p=2, dim=1)

    return box_features


@torch.no_grad()
def collect_features_on_loader(model, data_loader):
    was_training = model.training
    model.eval()

    all_feats = []

    for batch in data_loader:
        feats = extract_gt_roi_features(model, batch)
        if feats is None or feats.numel() == 0:
            continue
        all_feats.append(feats.detach().cpu())

    if was_training:
        model.train()

    if len(all_feats) == 0:
        return None

    return torch.cat(all_feats, dim=0)

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        n_samples = total.size(0)
        bandwidth = l2_distance.detach().sum() / (n_samples * (n_samples - 1) + 1e-6)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / (bw + 1e-6)) for bw in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s = source.size(0)
    n_t = target.size(0)

    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    XX = kernels[:n_s, :n_s]
    YY = kernels[n_s:, n_s:]
    XY = kernels[:n_s, n_s:]
    YX = kernels[n_s:, :n_s]

    return XX.mean() + YY.mean() - XY.mean() - YX.mean()


class MMDMapper:
    def __init__(self, cfg):
        self.img_format = cfg.INPUT.FORMAT

        # 用测试阶段的固定变换，不用训练增强
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(min_size, max_size, sample_style)
        ])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)),
            dtype=torch.float32
        )

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances

        return dataset_dict

import random
from detectron2.data import DatasetCatalog

def build_fixed_subset_all_dicts(dataset_name, num_images=None, seed=1234):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    total = len(dataset_dicts)

    # num_images=None 表示使用全部图片
    if num_images is None:
        return dataset_dicts

    if total <= num_images:
        return dataset_dicts

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:num_images]

    subset_dicts = [dataset_dicts[i] for i in indices]
    return subset_dicts



def build_fixed_subset_dicts(dataset_name, num_images=150, seed=1234):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    total = len(dataset_dicts)

    if total <= num_images:
        return dataset_dicts

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:num_images]

    subset_dicts = [dataset_dicts[i] for i in indices]
    return subset_dicts


from torch.utils.data import Dataset, DataLoader
from detectron2.data.common import MapDataset, DatasetFromList
def mmd_batch_collator(batch):
    return batch


def build_mmd_loader_from_dicts(cfg, dataset_dicts, batch_size=1, num_workers=0):
    mapper = MMDMapper(cfg)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mmd_batch_collator,
    )
    return data_loader
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def sample_features(feats, max_points=2000, seed=42):
    """
    feats: Tensor [N, D] or numpy [N, D]
    """
    if feats is None:
        return None

    if torch.is_tensor(feats):
        feats = feats.detach().cpu().numpy()

    n = feats.shape[0]
    if n <= max_points:
        return feats

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return feats[idx]


def reduce_features(src_feats, tgt_feats, method="tsne", random_state=42):
    """
    src_feats: numpy [Ns, D]
    tgt_feats: numpy [Nt, D]
    return:
        src_2d: [Ns, 2]
        tgt_2d: [Nt, 2]
    """
    X = np.concatenate([src_feats, tgt_feats], axis=0)
    n_src = src_feats.shape[0]

    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)

    elif method.lower() == "tsne":
        # t-SNE 前先做一次 PCA 到 50 维，通常更稳
        if X.shape[1] > 50:
            X = PCA(n_components=50).fit_transform(X)

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (X.shape[0] - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        X_2d = reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown method: {method}")

    src_2d = X_2d[:n_src]
    tgt_2d = X_2d[n_src:]
    return src_2d, tgt_2d


def plot_domain_scatter(
    src_2d,
    tgt_2d,
    save_path,
    title="GT ROI Feature Distribution",
    src_label="Source",
    tgt_label="Target",
):
    plt.figure(figsize=(10, 10))  # 画布也稍微放大

    plt.scatter(
        src_2d[:, 0], src_2d[:, 1],
        s=10, alpha=0.6, label=src_label
    )
    plt.scatter(
        tgt_2d[:, 0], tgt_2d[:, 1],
        s=10, alpha=0.6, label=tgt_label
    )

    # ⭐ 重点：字体控制
    plt.title(title, fontsize=25)          # 标题
    plt.xlabel("Dim 1", fontsize=20)       # x轴标签
    plt.ylabel("Dim 2", fontsize=20)       # y轴标签

    plt.xticks(fontsize=18)                # x轴刻度
    plt.yticks(fontsize=18)                # y轴刻度

    plt.legend(fontsize=24, markerscale=3.0)  # 图例（你要的重点）

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)        # dpi也建议提高
    plt.close()

    print(f"[VIS] saved scatter plot to: {save_path}")



def visualize_fixed_subset_features(
    cfg,
    model,
    source_dataset_name,
    target_dataset_name,
    output_dir,
    num_images=None,
    batch_size=1,
    num_workers=0,
    seed=1234,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 固定子集
    source_dicts = build_fixed_subset_all_dicts(
        source_dataset_name, num_images=200, seed=seed
    )
    target_dicts = build_fixed_subset_all_dicts(
        target_dataset_name, num_images=167, seed=seed
    )

    # 2. dataloader
    source_loader = build_mmd_loader_from_dicts(
        cfg, source_dicts, batch_size=batch_size, num_workers=num_workers
    )
    target_loader = build_mmd_loader_from_dicts(
        cfg, target_dicts, batch_size=batch_size, num_workers=num_workers
    )

    # 3. 提特征
    src_feats = collect_features_on_loader(model, source_loader)
    tgt_feats = collect_features_on_loader(model, target_loader)

    if src_feats is None or tgt_feats is None:
        print("[VIS] source or target features are None, skip visualization.")
        return

    print(f"[VIS] src_feats shape: {tuple(src_feats.shape)}")
    print(f"[VIS] tgt_feats shape: {tuple(tgt_feats.shape)}")

    # 4. 保存原始特征
    torch.save(
        {
            "src_feats": src_feats.cpu(),
            "tgt_feats": tgt_feats.cpu(),
        },
        os.path.join(output_dir, "gt_roi_features.pt")
    )

    if torch.is_tensor(src_feats):
        src_feats_np = src_feats.detach().cpu().numpy()
    else:
        src_feats_np = src_feats

    if torch.is_tensor(tgt_feats):
        tgt_feats_np = tgt_feats.detach().cpu().numpy()
    else:
        tgt_feats_np = tgt_feats


    # # 5. 采样，避免点太多图太糊
    # src_feats_np = sample_features(src_feats, max_points=max_points, seed=seed)
    # tgt_feats_np = sample_features(tgt_feats, max_points=max_points, seed=seed)

    # 6. PCA 可视化
    src_pca, tgt_pca = reduce_features(src_feats_np, tgt_feats_np, method="pca")
    plot_domain_scatter(
        src_pca,
        tgt_pca,
        save_path=os.path.join(output_dir, "gt_roi_pca.png"),
        title="Drop Feature Distribution",
        src_label="Source",
        tgt_label="Target",
    )

    # 7. t-SNE 可视化
    src_tsne, tgt_tsne = reduce_features(src_feats_np, tgt_feats_np, method="tsne")
    plot_domain_scatter(
        src_tsne,
        tgt_tsne,
        save_path=os.path.join(output_dir, "gt_roi_tsne.png"),
        title="DropFeature Distribution",
        src_label="Source",
        tgt_label="Target",
    )

    print("[VIS] done.")


# Supervised-only Trainer基线训练模型

class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]


# train_loop 方法是训练过程的主循环。start_iter 和 max_iter 分别表示训练的起始迭代次数和最大迭代次数。
    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()


# run_step 方法完成了训练中的每一个步骤：
# 获取训练数据并计算数据加载时间。
# 进行前向传播，计算损失和度量。
# 计算地面真值的边界框数量。
# 选择损失项并计算总损失。
# 记录度量数据和损失。
# 执行反向传播并更新模型参数。


    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


# 为数据集选择合适的评估标准
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=200))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
    
def l2_squared_loss( pred_boxes, true_boxes):
    # 计算两组边界框之间的 L2 平方距离
    squared_loss = torch.sum((pred_boxes - true_boxes) ** 2, dim=1).mean()
    return squared_loss    

def iou(box1, box2):
    """
    计算两个框之间的 IOU (Intersection over Union)
    box1, box2: [xmin, ymin, xmax, ymax]
    """
    device = box1.device  # 获取设备（如CUDA）
    
    # 计算交集的坐标
    xmin = torch.max(box1[0], box2[0])
    ymin = torch.max(box1[1], box2[1])
    xmax = torch.min(box1[2], box2[2])
    ymax = torch.min(box1[3], box2[3])
    
    # 计算交集区域的面积
    intersection_area = torch.max(xmax - xmin, torch.tensor(0.0, device=device)) * torch.max(ymax - ymin, torch.tensor(0.0, device=device))
    
    # 计算每个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集的面积
    union_area = area1 + area2 - intersection_area
    
    # 计算 IOU
    return intersection_area / union_area

def compute_iou_loss(pred_instances_k, pred_instances_q):
    """
    计算 pred_instances_k 和 pred_instances_q 中每对框的 IOU损失
    这里要求的是同一行框之间的 IOU损失
    """
    iou_losses = []
    device = pred_instances_k.device
    # 逐行计算 IOU 损失
    for box_k, box_q in zip(pred_instances_k, pred_instances_q):
        iou_score = iou(box_k, box_q)
        iou_loss = 1 - iou_score  # IOU损失是 1 - IOU
        iou_losses.append(iou_loss)
    
    return torch.tensor(iou_losses, device=device)

def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似性
    """
    a_normalized = F.normalize(a, p=2, dim=-1)  # L2 归一化
    b_normalized = F.normalize(b, p=2, dim=-1)
    return torch.sum(a_normalized * b_normalized, dim=-1)  # 返回余弦相似性


def filter_boxes_by_scores_sigmoid(instances, thre=0.9, m=0.5, k=10):
    """
    直接采用Sigmoid放大后筛选
    Args:
        instances: 包含 scores (tensor) 和 gt_boxes.tensor 的对象
        thre: 阈值（基于Sigmoid映射后的分数）
        m: Sigmoid中点
        k: Sigmoid曲线陡峭度
    Return:
        filtered_boxes: 筛选后的gt_boxes
        filtered_scores: 对应的映射后得分
    """

    scores = instances.scores  # 原始得分，例如范围 0~0.8
    
    # Sigmoid 映射：S(x) = 1 / (1 + e^{-k(x - m)})
    sigmoid_scores = 1 / (1 + torch.exp(-k * (scores - m)))
    
    # 筛选条件
    mask = sigmoid_scores > thre
    
    filtered_boxes = instances.gt_boxes.tensor[mask]
    filtered_scores = sigmoid_scores[mask]

    return filtered_boxes, filtered_scores



def find_indices(t_boxes, t_low_boxes):
    """
    找到 t_low_boxes 中每个框在 t_boxes 中完全相同的索引
    """
    indices = []
    # 遍历 t_low_boxes 中的每个框
    for low_box in t_low_boxes.tensor:
        # 查找与 low_box 完全相同的框
        match = torch.all(t_boxes.tensor == low_box, dim=1)  # 对比每个框
        idx = torch.nonzero(match, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:  # 如果找到了匹配的框
            indices.append(idx.item())  # 取出索引
    return torch.tensor(indices, dtype=torch.long)

def compute_cosine_similarity_matrix(predictions_k, predictions_q):
    """
    计算 predictions_k 和 predictions_q 中每对向量之间的余弦相似性
    """
    cosine_similarities = []

    # 逐行计算余弦相似性
    for k_row, q_row in zip(predictions_k, predictions_q):
        similarity = cosine_similarity(k_row, q_row)
        cosine_similarities.append(similarity)
    
    # 返回余弦相似性矩阵
    return torch.stack(cosine_similarities)
   
def _kl_divergence(weak_logits, strong_logits, weight=None, reduction='mean'):
    """
    weak_logits: Tensor of shape [n, num_classes]
    strong_logits: Tensor of shape [n, num_classes]
    weight: Optional Tensor of shape [n]
    reduction: 'mean' or 'sum'
    """
    # Convert logits to probabilities
    weak_probs = F.softmax(weak_logits, dim=1)
    strong_log_probs = F.log_softmax(strong_logits, dim=1)
    
    # Compute KL divergence per sample (no reduction yet)
    kl_per_sample = F.kl_div(strong_log_probs, weak_probs, reduction='none')  # shape: [n, num_classes]
    kl_per_sample = kl_per_sample.sum(dim=1)  # shape: [n]

    # Apply weights if provided
    if weight is not None:
        kl_per_sample = kl_per_sample * weight

    # Final reduction
    if reduction == 'mean':
        loss = kl_per_sample.sum() / (weight.sum() if weight is not None else kl_per_sample.shape[0])
    elif reduction == 'sum':
        loss = kl_per_sample.sum()
    else:
        loss = kl_per_sample  # no reduction

    return loss


from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class GTTestMapper:
    def __init__(self, cfg):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train=False)
        self.img_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # 按测试增强对图像 + GT 同步变换
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.tfm_gens)(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2,0,1)), dtype=torch.float32
        )

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances

        return dataset_dict


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        if comm.is_main_process():
            print_detectron2_params(model)


        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        # ===== MMD fixed subset loaders =====
        source_dataset_name = cfg.DATASETS.TRAIN_LABEL[0]
        target_dataset_name = cfg.DATASETS.TRAIN_UNLABEL[0]

        self.mmd_source_dicts = build_fixed_subset_dicts(
            source_dataset_name, num_images=150, seed=1234
        )
        self.mmd_target_dicts = build_fixed_subset_dicts(
            target_dataset_name, num_images=150, seed=5678
        )

        self.mmd_source_loader = build_mmd_loader_from_dicts(
            cfg, self.mmd_source_dicts, batch_size=1, num_workers=0
        )
        self.mmd_target_loader = build_mmd_loader_from_dicts(
            cfg, self.mmd_target_dicts, batch_size=1, num_workers=0
        )

        self.mmd_period = 1000   # 或者你改成按 epoch 触发

        # 检查是否是分布式训练（即 worker 数量大于 1），如果是，就用 DistributedDataParallel（DDP）包裹学生模型，这样可以在多台机器上并行训练模型。
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

# 初始化检查点工具（Checkpointer）：负责模型的保存和恢复。
# 设置训练的起始和最大迭代次数：为训练过程设定迭代的范围。
# 保存配置文件：将传入的配置文件保存以便在训练过程中使用。
# 初始化探针（Probe）：用于监控训练过程。
# 注册钩子：通过钩子在训练过程中插入特定的操作（如保存模型、记录日志等）



        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        #self.start_iter = 7499
        self.start_iter = 1
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks()) 


    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

       
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)   #目标域数据集经过两种强弱增强的方式生成两种类型的数据
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)
    

#这段代码说明了无论是带标签的数据集还是没有标签的数据集都会被经过强弱不同的增强方式来进行数据的预处理并引入数据加载器中


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    # 定义了一个阈值筛选函数
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    @torch.no_grad()
    def evaluate_fixed_subset_mmd(self):
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        device = next(model.parameters()).device

        source_feats = collect_features_on_loader(model, self.mmd_source_loader)
        target_feats = collect_features_on_loader(model, self.mmd_target_loader)

        if source_feats is None or target_feats is None:
            return None

        source_feats = source_feats.to(device)
        target_feats = target_feats.to(device)

        mmd_value = mmd_rbf(source_feats, target_feats)

        return {
            "mmd/fixed200_gt_roi": float(mmd_value.item()),
            "mmd/source_num_boxes": float(source_feats.size(0)),
            "mmd/target_num_boxes": float(target_feats.size(0)),
        }


    def threslow_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits <= thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores <= thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_low_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threslow_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output    


    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    




    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================


# run_step_full_semisup 的方法。它实现了一个半监督学习（semi-supervised learning）训练步骤，具体来说，涉及模型的前向传播、损失计算和教师模型更新。


    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak


    # label_data_q：强增强（strong augmentation）后的带标签数据。
    # label_data_k：弱增强（weak augmentation）后的带标签数据。
    # unlabel_data_q：强增强后的无标签数据。
    # unlabel_data_k：弱增强后的无标签数据。

        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
# 判断当前迭代数是否小于配置文件中的 BURN_UP_STEP，即判断是否处于“burn-in”阶段。在这个阶段，模型仅进行带标签数据的监督训练。
        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

# 如果当前迭代数大于或等于 BURN_UP_STEP，则进入后续的半监督学习阶段。
# 如果当前迭代数等于 BURN_UP_STEP，则更新教师模型，self._update_teacher_model(keep_rate=0.00) 可能是通过某种方式更新教师模型的参数（例如，使用 EMA 或其他方法），并设置 keep_rate 为 0，表示完全更新教师模型。

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model教师模型会完全复制学生模型的参数，keep_rate=0.00 表示完全用学生模型替代教师模型。
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()
#             如果当前迭代数与 BURN_UP_STEP 的差值能被 TEACHER_UPDATE_ITER 整除，则根据配置的 EMA_KEEP_RATE 更新教师模型。
# EMA_KEEP_RATE 可能表示教师模型更新时的指数平均保留率。
            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 
            gt_unlabel_k = self.get_label(unlabel_data_k)   #从弱增强（unlabel_data_k）的无标签数据中提取标签。
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)
            

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)#从强增强的无标签数据（unlabel_data_q）中删除标签
            unlabel_data_k = self.remove_label(unlabel_data_k)#从弱增强的无标签数据（unlabel_data_k）中删除标签

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():  #教师模型不进行反向传播
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

#             功能: 这行代码调用教师模型（model_teacher）对无标签数据（unlabel_data_k）进行前向推理，生成伪标签。branch="unsup_data_weak" 表示使用弱增强的无标签数据进行推理。
#             proposals_rpn_unsup_k 和 proposals_roih_unsup_k 是教师模型对无标签数据生成的候选框。通常，proposals_rpn_unsup_k 是通过 RPN（区域提议网络）生成的提议框，而 proposals_roih_unsup_k 是通过 ROI 头生成的提议框。
#              _ 用于占位，表示我们不关心模型输出的其他部分。

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']  
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
# 阈值用于筛选提议框。在生成伪标签时，只有那些置信度高于该阈值的提议框才会被保留下来，作为伪标签。
            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            #Process pseudo labels and thresholding     proposals_rpn_unsup_k 是教师模型生成的无标签数据的 RPN 提议框。
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
                
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            # 通过调用 self.process_pseudo_label 函数，生成并筛选伪标签提议框：
            # proposals_rpn_unsup_k 是原始的 RPN 提议框。
            # cur_threshold 是用于筛选提议框的阈值。
            # "rpn" 指明是针对 RPN 提议框进行处理。
            # "thresholding" 表示采用阈值方法进行伪标签筛选。
            # 函数 process_pseudo_label 返回两个值：
            # pesudo_proposals_rpn_unsup_k：经过阈值筛选后得到的伪标签提议框。
            # nun_pseudo_bbox_rpn：伪标签提议框的数量。

            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)
            
            
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k  #筛选后的rpn提议框伪标签
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )

            

            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k  #筛选后的roihead的提议框伪标签包括了分类和候选框的回归：最终的伪标签
            
            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            
            label_k = label_data_k

            all_label_data = label_data_q + label_data_k   #源域带标签的增强后的两种数据集合并
            all_unlabel_data = unlabel_data_q

            # 4. input both strongly and weakly augmented labeled data into student model
            # 监督学习部分，包括源域图像和增强后的源域图像
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            # 5. input strongly augmented unlabeled data into student model
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data
            #域分类器对抗学习部分
            
            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_data_k
            #all_domain_data = label_data_k + unlabel_data_k
            record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)
            record_all_insdomain_data, _, _, _ = self.model(all_domain_data, branch="ins_domain")
            record_dict.update(record_all_insdomain_data)

            #引入学生网络对比损失
            predictions_k, pred_instances_k, filter_proposals_rpn_k = self.model(label_k, branch="ex_label_weak")
            
            predictions_q, pred_instances_q = self.model(label_data_q, branch="ex_label_strong", given_proposals=filter_proposals_rpn_k)
            cosine_similarity_matrix = compute_cosine_similarity_matrix(predictions_k[0], predictions_q[0])
                        #ca_loss = _kl_divergence(predictions_k[0], predictions_q[0])
                        #re_loss = l2_squared_loss(pred_instances_q, pred_instances_k)/pred_instances_k.size(0)   
            iou_loss = compute_iou_loss(pred_instances_k, pred_instances_q)
            ca_loss = (iou_loss * (1-cosine_similarity_matrix)).mean()
            losses = {}
            losses["loss_ca"] = ca_loss*0.04
                         #losses["loss_re"] = re_loss*0.04  #0.1            
            record_dict.update(losses)
            
            #  引入LPL损失
            t_proposal_features_norm, t_proposal_features_roih_logits, t_predict_boxes, teacher_proposals = self.model_teacher(unlabel_data_k, branch="ex_unlabel_weak", given_proposals=proposals_rpn_unsup_k)
               
            s_proposal_features_norm,s_proposal_features_roih_logits = self.model_teacher(unlabel_data_q, branch="ex_unlabel_strong", given_proposals=proposals_rpn_unsup_k)

            pesudo_proposals_roih_low_unsup_k, _ = self.process_pseudo_low_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            pesudo_proposals_roih_high_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )            
            t_low_boxes, _ = filter_boxes_by_scores_sigmoid(pesudo_proposals_roih_low_unsup_k[0])
            
            t_boxes = Boxes(t_predict_boxes) 
            t_low_boxes= Boxes(pesudo_proposals_roih_low_unsup_k[0].gt_boxes.tensor)  #教师模型输出的0.8阈值roi_head过滤后的低置信度伪标签结果，roih_box            
            t_high_boxes= Boxes(pesudo_proposals_roih_high_unsup_k[0].gt_boxes.tensor)
            t_expend_boxes = Boxes(torch.empty((0, 4)))
            topk = 1            
            if t_low_boxes.tensor.numel() == 0 and t_high_boxes.tensor.numel() == 0 and pesudo_proposals_roih_low_unsup_k[0].scores.numel() > 0:    
                topk_indices = torch.topk(pesudo_proposals_roih_low_unsup_k[0].scores, k=topk).indices
                topk_boxes = pesudo_proposals_roih_low_unsup_k[0].pred_boxes.tensor[topk_indices]
                t_expend_boxes = Boxes(topk_boxes)                           
            elif t_low_boxes.tensor.numel() == 0 and t_high_boxes.tensor.numel() == 0 and pesudo_proposals_roih_low_unsup_k[0].scores.numel() == 0:
                topk_indices = torch.topk(proposals_rpn_unsup_k[0].objectness_logits, k=topk).indices

                topk_rpn = proposals_rpn_unsup_k[0].proposal_boxes.tensor[topk_indices]
                topk_rpn_logits = t_proposal_features_roih_logits[1][topk_indices]
                box2box_transform = Box2BoxTransform(weights=(10.0, 10.0, 5.0, 5.0))
                topk_boxes = box2box_transform.apply_deltas(topk_rpn_logits, topk_rpn)
                t_expend_boxes = Boxes(topk_boxes)
            device = t_high_boxes.tensor.device
            t_expend_boxes = Boxes(t_expend_boxes.tensor.to(device))
  
            result_boxes = Boxes.cat([t_high_boxes, t_low_boxes, t_expend_boxes])            
            
            iou_matrix = pairwise_iou(t_boxes, result_boxes)
            #c_similarity = F.cosine_similarity(s_proposal_features_norm.detach(), t_proposal_features_norm.detach(), dim=1)#强弱两组无标签图像的提议框特征分布的余弦相似度
            cosine_similarity_matrix = 1- compute_cosine_similarity_matrix(s_proposal_features_norm.detach(), t_proposal_features_norm.detach())

                
                # if iou_matrix.shape[1] == 0:
                #     print("没有iou交集")

            t_indices = torch.nonzero((torch.max(iou_matrix, dim=1).values > 0.5)).flatten()#0.6
            weight = cosine_similarity_matrix[t_indices]
#                 if t_indices_filtered.nelement() == 0:


            loses = {}
            loses["loss_lpl"] = _kl_divergence(s_proposal_features_roih_logits[0][t_indices],
                                                    t_proposal_features_roih_logits[0][t_indices],
                                                    weight=weight, reduction='mean') * 0.1  # shape: [n]
            lpl_loss = loses.get("loss_lpl")
            if lpl_loss is not None:
                if not hasattr(self, "optimizer_teacher"):
                    self.optimizer_teacher = torch.optim.SGD(
                        self.model_teacher.parameters(),
                        lr=self.cfg.SOLVER.BASE_LR * 0.1,  # 自更新更温和
                        momentum=0.9,
                        weight_decay=0.0001
                    )
                record_dict["loss_lpl_teacher"] = lpl_loss.detach()
                self.optimizer_teacher.zero_grad()
                lpl_loss.backward()
                self.optimizer_teacher.step()    
            


            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0

                        # 如果损失项是 "loss_rpn_loc_pseudo" 或 "loss_box_reg_pseudo"，这表示伪标签回归损失。在训练过程中，伪标签回归的损失项应该被设置为 0，即 不参与梯度更新。
                        # 通过将该损失乘以 0，使其对总损失没有贡献。
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    # elif (
                    #     key == "loss_D_img_s" or key == "loss_D_img_t"
                    # ):  # set weight for discriminator
                    #     # import pdb
                    #     # pdb.set_trace()
                    #     loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                    
                    elif key == "loss_D_ins_s" :
                        
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_LOSS_WEIGHT #Need to modify defaults and yaml                    
                    
                    elif key == "loss_D_ins_t":
                        
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.INS_LOSS_WEIGHT #Need to modify defaults and yaml                    

                    # elif key == "loss_lpl":
                       
                    #     loss_dict[key] = record_dict[key] * 0.1 #Need to modify defaults and yaml

                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

     

        metrics_dict = record_dict 
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
        if comm.is_main_process() and self.iter > 0 and self.iter % self.mmd_period == 0:
            mmd_metrics = self.evaluate_fixed_subset_mmd()
            if mmd_metrics is not None:
                for k, v in mmd_metrics.items():
                    self.storage.put_scalar(k, v)
        

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                ) #student_model_dict[key] * (1 - keep_rate) + value * keep_rate：这是更新教师模型的关键步骤。采用了指数加权平均的方法更新教师模型
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())




    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)#导入测试数据集以及标签

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,   #测试周期
                self.model,             
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,       
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD    
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
                
            _last_eval_results_teacher = {
                k + "_teacher": self._last_eval_results_teacher[k]
                for k in self._last_eval_results_teacher.keys()
            }
            
            return _last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))
        

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=200))
        return ret
