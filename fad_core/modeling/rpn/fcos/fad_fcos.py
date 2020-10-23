# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from torch import nn

from fcos_core.modeling.rpn.fcos.inference import make_fcos_postprocessor
from fcos_core.modeling.rpn.fcos.loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

class FADFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FADFCOSHead, self).__init__()
        
        self.cfg = cfg
        self.fpn_lvl = 5

        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        # import fad modules
        from fad_core.modeling.modules.search_rcnn import SearchRCNN
        from fad_core.modeling.modules.augment_rcnn import AugmentRCNN

        # --------- cls tower            
        if cfg.MODEL.FAD.CLSTOWER:
            if cfg.MODEL.FAD.SEARCH:
                cls_tower = SearchRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_CLS, cfg.MODEL.FAD.NUM_CELLS_CLS, n_nodes=cfg.MODEL.FAD.NUM_NODES_CLS) 
            else:
                # augment 
                cls_tower = AugmentRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_CLS, cfg.MODEL.FAD.NUM_CELLS_CLS, cfg.MODEL.FAD.GENO_CLS[0]) 

        if cfg.MODEL.FAD.BOXTOWER:
            if cfg.MODEL.FAD.SEARCH:
                bbox_tower = SearchRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_BOX, cfg.MODEL.FAD.NUM_CELLS_BOX, n_nodes=cfg.MODEL.FAD.NUM_NODES_BOX) 
            else:        
                # augment 
                bbox_tower = AugmentRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_BOX, cfg.MODEL.FAD.NUM_CELLS_BOX, cfg.MODEL.FAD.GENO_BOX[0]) 
     
        if cfg.MODEL.FAD.CLSTOWER:
            self.add_module('cls_tower', cls_tower)
        else:
            self.add_module('cls_tower', nn.Sequential(*cls_tower))

        if cfg.MODEL.FAD.BOXTOWER:
            self.add_module('bbox_tower', bbox_tower)
        else:
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # ---- if 1x1 conv to reduce dim first
        if cfg.MODEL.FAD.CLSTOWER:
            self.cls_reduce = nn.Conv2d(
                in_channels*cfg.MODEL.FAD.NUM_NODES_CLS, in_channels, kernel_size=1, stride=1,
                padding=0
            ) 

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        if cfg.MODEL.FAD.BOXTOWER:
            self.bbox_pred = nn.Conv2d(
                in_channels*min(1,cfg.MODEL.FAD.NUM_NODES_BOX), 4, kernel_size=3, stride=1,
                padding=1
            )
            self.box_reduce = nn.Conv2d(
                in_channels*cfg.MODEL.FAD.NUM_NODES_BOX, in_channels, kernel_size=1, stride=1,
                padding=0
            ) 
        else:
            self.bbox_pred = nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1,
                padding=1
            )

        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        list_init = [self.cls_logits, self.bbox_pred, self.centerness]
        if self.cfg.MODEL.FAD.CLSTOWER:
             list_init.append(self.cls_reduce)
        else:
             list_init.append(self.cls_tower)
        if not self.cfg.MODEL.FAD.BOXTOWER:
             list_init.append(self.bbox_tower)
             
        for modules in list_init:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x, weights_normal=None):
        logits = []
        bbox_reg = []
        centerness = []

        n_nodes = self.cfg.MODEL.FAD.NUM_NODES_BOX if self.cfg.MODEL.FAD.BOXTOWER else self.cfg.MODEL.FAD.NUM_NODES_CLS

        # check the number of modules to be searched
        if self.cfg.MODEL.FAD.SEARCH:
            n_module = len(weights_normal) // n_nodes

        for l, feature in enumerate(x):
            # ----------- bbox stream 
            if not self.cfg.MODEL.FAD.BOXTOWER:
                bbox_feature = self.bbox_tower(feature)
                bbox_pred = self.scales[l](self.bbox_pred(bbox_feature))
                if self.norm_reg_targets:
                    if self.training:
                        bbox_pred = F.relu(bbox_pred)
                        bbox_reg.append(bbox_pred)
                    else:
                        bbox_reg.append(bbox_pred * self.fpn_strides[l])
                else:
                    bbox_reg.append(torch.exp(bbox_pred))
            else:
                if self.cfg.MODEL.FAD.SEARCH:
                    bbox_feature = self.box_reduce(self.bbox_tower(feature, weights_normal[:n_nodes]))
                    bbox_pred = self.scales[l](self.bbox_pred(bbox_feature))
                    if self.norm_reg_targets:
                        if self.training:
                            bbox_pred = F.relu(bbox_pred)
                            bbox_reg.append(bbox_pred)
                        else:
                            bbox_reg.append(bbox_pred * self.fpn_strides[l])
                    else:
                        bbox_reg.append(torch.exp(bbox_pred))
                else: # augment
                    bbox_feature = self.box_reduce(self.bbox_tower(feature))
                    bbox_pred = self.scales[l](self.bbox_pred(bbox_feature))
                    if self.norm_reg_targets:
                        if self.training:
                            bbox_pred = F.relu(bbox_pred)
                            bbox_reg.append(bbox_pred)
                        else:
                            bbox_reg.append(bbox_pred * self.fpn_strides[l])
                    else:
                        bbox_reg.append(torch.exp(bbox_pred))                       

            # ---------- cls tower
            if not self.cfg.MODEL.FAD.CLSTOWER:
                cls_tower = self.cls_tower(feature)
                logits.append(self.cls_logits(cls_tower))
            else:
                if self.cfg.MODEL.FAD.SEARCH:
                    if n_module == 1: # search for cls only 
                        cls_tower = self.cls_tower(feature, weights_normal)
                    elif n_module == 2: 
                        # search for both
                        cls_tower = self.cls_tower([bbox_feature, feature], weights_normal[n_nodes:])
                    else:
                        pdb.set_trace()
                else: # augment
                    cls_tower = self.cls_tower([bbox_feature, feature]) 
                logits.append(self.cls_logits(self.cls_reduce(cls_tower)))

            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_feature))
            else:
                centerness.append(self.centerness(cls_tower))            

        return logits, bbox_reg, centerness


class FADFCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FADFCOSModule, self).__init__()

        head = FADFCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.cfg = cfg

    def forward(self, images, features, targets=None, weights_normal=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        if self.cfg.MODEL.FAD.SEARCH:
            box_cls, box_regression, centerness = self.head(features, weights_normal)
        else:
            box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fad_fcos(cfg, in_channels):
    return FADFCOSModule(cfg, in_channels)
