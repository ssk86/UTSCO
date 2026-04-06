# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList
import math
from detectron2.layers import cat
import numpy as np
import torch
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def convert_image_to_rgb(image_tensor, input_format):
    """
    Converts a tensor image to RGB format.

    Args:
        image_tensor (Tensor): A tensor image of shape (C, H, W).
        input_format (str): The input format, such as 'BGR', 'RGB', 'GRAY', etc.

    Returns:
        np.ndarray: An RGB image in numpy format with shape (H, W, C).
    """
    # Ensure the image is in the correct format (C, H, W) to (H, W, C)
    if isinstance(image_tensor, torch.Tensor):
        # Convert the tensor from (C, H, W) to (H, W, C)
        image = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError("Input should be a torch.Tensor")

    # Handle different input formats
    if input_format == 'BGR':
        # Convert from BGR to RGB
        image = image[..., ::-1]  # Reverse the color channels (BGR -> RGB)
    elif input_format == 'GRAY':
        # Convert from grayscale to RGB by duplicating the single channel
        image = np.stack([image] * 3, axis=-1)
    elif input_format == 'RGBA':
        # Optionally, remove the alpha channel and keep only RGB
        image = image[..., :3]  # Only take the first 3 channels (RGB)
    
    else:
        # If the format is already RGB, we do nothing
        pass
    
    # Ensure that the values are in the range [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image
class FCDiscriminator_ins(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128, num_out_classes=1):
        super(FCDiscriminator_ins, self).__init__()

        # 卷积层部分
        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)  # 输入是特征图的通道数3
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)

        # 全局池化层，用于将每个7x7特征图池化为1维特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 输出大小为1x1

        # 分类头，最后的全连接层，用于分类（源域/目标域）
        self.fc = nn.Linear(ndf2, num_out_classes)  # ndf2是通道数，num_out_classes是分类类别数（2类）

        # LeakyReLU 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 卷积层提取特征
        # print(f'Input shape: {x.shape}')  # 打印输入形状
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')  # 打印 conv1 后的形状
        x = self.leaky_relu(x)
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')  # 打印 conv2 后的形状
        x = self.leaky_relu(x)
        x = self.conv3(x)
        # print(f'After conv3: {x.shape}')  # 打印 conv3 后的形状
        x = self.leaky_relu(x)

        # 全局平均池化，将每个特征图池化成1维
        x = self.global_pool(x)
        # print(f'After global_pool: {x.shape}')  # 打印池化后的形状

        # 展平池化后的特征图，准备输入到全连接层
        x = x.view(x.size(0), -1)  # 展平为 (batch_size * num_proposals, ndf2)
        # print(f'After view: {x.shape}')  # 打印展平后的形状

        # 通过全连接层进行域分类
        x = self.fc(x)

        return x  # logits 直接输出

    # def forward(self, x):
    #     # 卷积层提取特征
    #     x = self.conv1(x)
    #     x = self.leaky_relu(x)
    #     x = self.conv2(x)
    #     x = self.leaky_relu(x)
    #     x = self.conv3(x)
    #     x = self.leaky_relu(x)

    #     # 全局平均池化，将每个7x7的特征图池化成1维
    #     x = self.global_pool(x)

    #     # 展平池化后的特征图，准备输入到全连接层
    #     x = x.view(x.size(0), -1)  # 展平为 (batch_size * num_proposals, ndf2)

    #     # 通过全连接层进行域分类
    #     x = self.fc(x)

    #     # 不再使用 softmax，因为 F.binary_cross_entropy_with_logits 已经处理了 sigmoid
    #     return x  # logits 直接输出
############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ 动态GRL,Gradient reverse function
# class DynamicGradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         weight = ctx.alpha
#         grad_output = grad_output.neg() * weight
#         return grad_output, None

# def dynamic_grad_reverse(x, alpha):
#     return DynamicGradReverse.apply(x, alpha)

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################
#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        tau_min: float,
        tau_max: float,
        tau_gamma: float,
        max_iter: int,  
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        self.D_ins = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device)# Need to know the channel
        self.D_ins = FCDiscriminator_ins(self.backbone._out_feature_channels[self.dis_type]).to(self.device)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_gamma = float(tau_gamma)
        self.max_iter = int(max_iter) 
        # self.bceLoss_func = nn.BCEWithLogitsLoss()
    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel
        self.D_ins = FCDiscriminator_ins(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,#域分类器作用res4层
            "tau_min": cfg.SEMISUPNET.TAU_MIN,
            "tau_max": cfg.SEMISUPNET.TAU_MAX,
            "tau_gamma": cfg.SEMISUPNET.TAU_GAMMA,
            "max_iter": cfg.SOLVER.MAX_ITER,
            # "dis_loss_ratio": cfg.xxx,可视化标准化部分参数未定义
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


    #标准化批处理设备迁移保证了输入的图像数据格式和范围的一致性
    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t
    
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
    



    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False 
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        cur_iter = int(getattr(self, "iter", 0))
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        # def get_dynamic_alpha(current_iters, max_iters, alpha_initial=0.1, alpha_final=1.0, use_exponential_growth=False):
        #     if use_exponential_growth:
        #         # 指数增长
        #         alpha = alpha_initial * (alpha_final / alpha_initial) ** (current_iters / max_iters)
        #     else:
        #         # 线性增长
        #         alpha = alpha_initial + (alpha_final - alpha_initial) * (current_iters / max_iters)
        #     return alpha

        # 域分类器模式，返回两个域的分类损失
        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # current_iters= iters
            
            # alpha = get_dynamic_alpha(current_iters ,25000 , alpha_initial=0.1, alpha_final=1.0, use_exponential_growth=False)

            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s*0.01
            losses["loss_D_img_t"] = loss_D_img_t*0.01            
            # losses["loss_D_img_s"] = loss_D_img_s*0.01
            # losses["loss_D_img_t"] = loss_D_img_t*0.01实例级实验所用权重
            return losses, [], [], None

        if branch == "ins_domain":
            # tau_t = self.tau_min + (self.tau_max - self.tau_min) * (cur_iter / self.max_iter) ** self.tau_gamma
            # tau_t = torch.tensor(tau_t)
            # tau_logit = torch.log(tau_t / (1 - tau_t))
            source_label = 0
            target_label = 1
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features_s = self.backbone(images_s.tensor)

            proposals_rpn_s, _ = self.proposal_generator(
                images_s, features_s, None, compute_loss=False
            )


            # filter_proposals_rpn_s, _ = self.process_pseudo_label(
            #     proposals_rpn_s, tau_logit , "rpn", "thresholding"
            # )
                
            proposal_boxes_s = [Boxes(getattr(instance, 'proposal_boxes').tensor) for instance in proposals_rpn_s]
            logits = [getattr(instance, 'objectness_logits') for instance in proposals_rpn_s]
            scores = torch.tensor([sigmoid(x) for x in logits[0]])
            #scores = [getattr(instance, 'scores') for instance in filter_proposals_rpn_s]    
            proposal_features_s = self.roi_heads._shared_roi_transform([features_s['res4']], proposal_boxes_s)
            p_features_s = grad_reverse(proposal_features_s)
            
            if p_features_s.size(0) > 0: 
                D_ins_out_s = self.D_ins(p_features_s) #F.binary_cross_entropy_with_logits 这个函数在内部已经包含了 sigmoid 操作
                # weights = torch.exp(torch.abs(scores - 1)).detach().unsqueeze(1).to(self.device)
                loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s, torch.FloatTensor(D_ins_out_s.data.size(0)).fill_(source_label).unsqueeze(1).to(self.device))
                #loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s, torch.FloatTensor(D_ins_out_s.data.size(0)).fill_(source_label).unsqueeze(1).to(self.device))
            else:
                # 如果 filter_proposals_rpn_t 为空，可以跳过后续的损失计算
                loss_D_ins_s = 0.0  # 不计算损失
            

            features_t = self.backbone(images_t.tensor)
            proposals_rpn_t, _ = self.proposal_generator(
                images_t, features_t, None, compute_loss=False
            )

            
            # filter_proposals_rpn_t, _ = self.process_pseudo_label(
            #     proposals_rpn_t, tau_logit, "rpn", "thresholding"
            # )
               
            proposal_boxes_t = [Boxes(getattr(instance, 'proposal_boxes').tensor) for instance in proposals_rpn_t]
            logits_t = [getattr(instance, 'objectness_logits') for instance in proposals_rpn_t]
            scores_t = torch.tensor([sigmoid(x) for x in logits_t[0]])
            #scores_t = [getattr(instance, 'scores') for instance in filter_proposals_rpn_t]  
            proposal_features_t = self.roi_heads._shared_roi_transform([features_t['res4']], proposal_boxes_t)            
            

            features_dt = grad_reverse(proposal_features_t)
            if features_dt.size(0) > 0:    
                D_ins_out_t = self.D_ins(features_dt)
                
                weights = torch.exp(torch.abs(scores_t - 1)).detach().unsqueeze(1).to(self.device)
                loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t, torch.FloatTensor(D_ins_out_t.data.size(0)).fill_(target_label).unsqueeze(1).to(self.device))
                #loss_D_ins_t = F.binary_cross_entropy_with_logits(D_ins_out_t, torch.FloatTensor(D_ins_out_t.data.size(0)).fill_(target_label).unsqueeze(1).to(self.device))
            else:
            # 如果 filter_proposals_rpn_t 为空，可以跳过后续的损失计算
                loss_D_ins_t = 0.0  # 不计算损失
            
            

            losses = {}
            losses["loss_D_ins_s"] = loss_D_ins_s*0.001
            losses["loss_D_ins_t"] = loss_D_ins_t*0.001
            return losses, [], [], None


        # self.D_img.eval()

    #教师模型提特征：传入weak增强的无标签数据
        if branch == "ex_unlabel_weak":
            images_k = self.preprocess_image(batched_inputs)
            # print(images_k.tensor.shape) 
            features = self.backbone(images_k.tensor)
            
            # 假设字典的名称是 output_dict
            # 获取 'res4' 键对应的值
            teacher_proposals = given_proposals

            t_proposal_box_features = self.roi_heads._shared_roi_transform([features['res4']], [given_proposals[0].proposal_boxes]) #given_proposals是教师网络rpn未前景置信度过滤的提议框 (2000,1024,7,7)
            t_proposal_features_mean = self.roi_heads.box_head(t_proposal_box_features) #(2000,1024)  2000个提议框的特征向量，维度为1024，为后续计算特征向量的余弦相似度做准备，这是弱增强图像提议框的2000个框的特征向量
            
            t_proposal_features_norm = F.normalize(t_proposal_features_mean, dim=1) #将上面的2000个提议框的特征向量归一化，使得特征向量的模值为1，以便于后面的计算余弦相似度
            t_proposal_features_roih_logits = self.roi_heads.box_predictor(t_proposal_features_mean) #t_proposal_features_roih_logits是一个列表。logits[0] 代表cls预测（2000，2）    logits[1] 边界框deltas值（2000，4）
            t_proposal_boxes = cat([p.proposal_boxes.tensor for p in given_proposals], dim=0) #得到教师模型rpn提议框的框的坐标，为后续得到偏移后的预测框做铺垫（2000，4）
            
            t_predict_boxes = self.roi_heads.box_predictor.box2box_transform.apply_deltas(t_proposal_features_roih_logits[1], t_proposal_boxes) #经过roih层回归后的预测框 也就是弱强混杂的伪标签的bbox，为后续的iou筛选弱伪标签作铺垫
            
            return t_proposal_features_norm, t_proposal_features_roih_logits, t_predict_boxes, teacher_proposals


    #教师模型提特征：传入q强增强的无标签数据
        if branch == "ex_unlabel_strong":
            images_q = self.preprocess_image(batched_inputs)
            features = self.backbone(images_q.tensor)
            s_proposal_box_features = self.roi_heads._shared_roi_transform([features['res4']], [given_proposals[0].proposal_boxes]) #given_proposals是教师网络rpn未前景置信度过滤的提议框（2000，1024，7，7）
            s_proposal_features_mean = self.roi_heads.box_head(s_proposal_box_features) #（2000，1024）
            s_proposal_features_norm = F.normalize(s_proposal_features_mean, dim=1) #（2000，1024）
            s_proposal_features_roih_logits = self.roi_heads.box_predictor(s_proposal_features_mean) # logits[0] 代表cls预测（2000，2）    logits[1] 边界框deltas值（2000，4）          


            return s_proposal_features_norm,s_proposal_features_roih_logits


        # label_data_k：弱增强（weak augmentation）后的带标签数据。
        if branch == "ex_label_weak":
            images_k = self.preprocess_image(batched_inputs)
            
            features = self.backbone(images_k.tensor)
            proposals_rpn_k, _ = self.proposal_generator(
                images_k, features, None, compute_loss=False
            )
            
                
                
                
            # filter_proposals_rpn_k, _ = self.process_pseudo_label(
            #     proposals_rpn_k, 0.0, "rpn", "thresholding"
            # )

            #filter_proposals_rpn_k = proposals_rpn_k
            proposal_boxes_k = [Boxes(getattr(instance, 'proposal_boxes').tensor) for instance in proposals_rpn_k] #proposal_boxes gt_boxes
            proposal_features_k = self.roi_heads._shared_roi_transform([features['res4']], proposal_boxes_k)
           
            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!                       
            box_features_k = self.roi_heads.box_head(proposal_features_k)
            predictions_k = self.roi_heads.box_predictor(box_features_k)        
            filter_proposals_boxes_k = cat([p.proposal_boxes.tensor for p in proposals_rpn_k], dim=0)
            pred_instances_k = self.roi_heads.box_predictor.box2box_transform.apply_deltas(predictions_k[1], filter_proposals_boxes_k)

            return predictions_k, pred_instances_k, proposals_rpn_k


        # label_data_q：强增强（strong augmentation）后的带标签数据。
        if branch == "ex_label_strong":
            images_q = self.preprocess_image(batched_inputs)
            features = self.backbone(images_q.tensor)
            filter_proposals_rpn_q = given_proposals
            proposal_boxes_q = [Boxes(getattr(instance, 'proposal_boxes').tensor) for instance in filter_proposals_rpn_q]
            proposal_features_q = self.roi_heads._shared_roi_transform([features['res4']], proposal_boxes_q)
            box_features_q = self.roi_heads.box_head(proposal_features_q)
            predictions_q = self.roi_heads.box_predictor(box_features_q)     
            filter_proposals_boxes_q = cat([p.proposal_boxes.tensor for p in filter_proposals_rpn_q], dim=0) #proposal_boxes
            pred_instances_q = self.roi_heads.box_predictor.box2box_transform.apply_deltas(predictions_q[1], filter_proposals_boxes_q)           
            
            return predictions_q, pred_instances_q

#  有监督损失分支
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
         
        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network，使用 Region Proposal Network（RPN）生成候选区域，并计算相关的损失
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch使用 RoI Heads 计算最终检测损失
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            boxes = [Boxes(getattr(instance, 'gt_boxes').tensor) for instance in gt_instances]
            
            ins_feature = self.roi_heads._shared_roi_transform([features['res4']], boxes)
            features_ins = grad_reverse(ins_feature)
            D_ins_out_s = self.D_ins(features_ins)
            loss_D_ins_s = F.binary_cross_entropy_with_logits(D_ins_out_s, torch.FloatTensor(D_ins_out_s.data.size(0)).fill_(source_label).unsqueeze(1).to(self.device))

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            # losses["loss_D_ins_s"] = loss_D_ins_s*0.001
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None



# 无监督弱分支 模式下进行训练或推理。这个分支的任务是输入图像没有任何标签，模型仅仅生成候选区域和预测结果。
# 教师网络生成伪标签
        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()
        

    # def visualize_training(self, batched_inputs, proposals, branch=""):
    #     """
    #     This function different from the original one:
    #     - it adds "branch" to the `vis_name`.

    #     A function used to visualize images and proposals. It shows ground truth
    #     bounding boxes on the original image and up to 20 predicted object
    #     proposals on the original image. Users can implement different
    #     visualization functions for different models.

    #     Args:
    #         batched_inputs (list): a list that contains input to the model.
    #         proposals (list): a list that contains predicted proposals. Both
    #             batched_inputs and proposals should have the same length.
    #     """
    #     from detectron2.utils.visualizer import Visualizer

    #     storage = get_event_storage()
    #     max_vis_prop = 10

    #     for input, prop in zip(batched_inputs, proposals):
    #         img = input["image"]
    #         img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
    #         v_gt = Visualizer(img, None)
    #         v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
    #         anno_img = v_gt.get_image()
    #         box_size = min(len(prop.proposal_boxes), max_vis_prop)
    #         v_pred = Visualizer(img, None)
    #         v_pred = v_pred.overlay_instances(
    #             boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
    #         )
    #         prop_img = v_pred.get_image()
    #         vis_img = np.concatenate((anno_img, prop_img), axis=1)
    #         vis_img = vis_img.transpose(2, 0, 1)
    #         vis_name = (
    #             "Left: GT bounding boxes "
    #             + branch
    #             + ";  Right: Predicted proposals "
    #             + branch
    #         )
    #         storage.put_image(vis_name, vis_img)
    #         break  # only visualize one image in a batch

def visualize_training(self, batched_inputs, proposals, branch=""):
    """
    A function used to visualize images and proposals. It shows ground truth
    bounding boxes on the original image and up to 20 predicted object
    proposals on the original image. Users can implement different
    visualization functions for different models.

    Args:
        batched_inputs (list): a list that contains input to the model.
        proposals (list): a list that contains predicted proposals. Both
            batched_inputs and proposals should have the same length.
    """
    from detectron2.utils.visualizer import Visualizer

    storage = get_event_storage()
    max_vis_prop = 10

    for input, prop in zip(batched_inputs, proposals):
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

        # Visualize ground truth if available
        if "instances" in input:  # If ground truth exists
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
        else:
            anno_img = img  # If no ground truth, just use the original image

        # Visualize proposals
        box_size = min(len(prop.proposal_boxes), max_vis_prop)
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
        )
        prop_img = v_pred.get_image()

        # Combine both images side by side
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        vis_img = vis_img.transpose(2, 0, 1)

        # Set visualization name
        vis_name = (
            "Left: GT bounding boxes "
            + branch
            + ";  Right: Predicted proposals "
            + branch
        )
        storage.put_image(vis_name, vis_img)
        break  # only visualize one image in a batch


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


