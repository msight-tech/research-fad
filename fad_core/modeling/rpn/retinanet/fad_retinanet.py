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

from fcos_core.modeling.rpn.retinanet.inference import  make_retinanet_postprocessor
from fcos_core.modeling.rpn.retinanet.loss import make_retinanet_loss_evaluator
from fcos_core.modeling.rpn.anchor_generator import make_anchor_generator_retinanet
from fcos_core.modeling.rpn.retinanet.retinanet import RetinaNetModule

from fcos_core.modeling.box_coder import BoxCoder

class FADRetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(FADRetinaNetHead, self).__init__()

        self.cfg = cfg         
        self.fpn_lvl = 5

        self.norm_reg_targets = False
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
                        * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.RETINANET.NUM_CONVS):
            if not cfg.MODEL.FAD.CLSTOWER:
                cls_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                cls_tower.append(nn.ReLU())
            if not cfg.MODEL.FAD.BOXTOWER:
                bbox_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                bbox_tower.append(nn.ReLU())

        # import FAD modules
        from fad_core.modeling.modules.search_rcnn import SearchRCNN
        from fad_core.modeling.modules.augment_rcnn import AugmentRCNN

        # --------- cls tower            
        if cfg.MODEL.FAD.CLSTOWER:
            if cfg.MODEL.FAD.SEARCH:
                cls_tower = SearchRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_CLS, cfg.MODEL.FAD.NUM_CELLS_CLS, n_nodes=cfg.MODEL.FAD.NUM_NODES_CLS) 
            else:
                # augment using searched Genotype
                cls_tower = AugmentRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_CLS, cfg.MODEL.FAD.NUM_CELLS_CLS, cfg.MODEL.FAD.GENO_CLS[0], cfg.MODEL.FAD.LOSS_MID_CLS, Cs=cfg.MODEL.FAD.CHANNEL_LIST_CLS, genotypeCH=cfg.MODEL.FAD.GENO_CHANNEL_CLS) 

        if cfg.MODEL.FAD.BOXTOWER:
            if cfg.MODEL.FAD.SEARCH:
                bbox_tower = SearchRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_BOX, cfg.MODEL.FAD.NUM_CELLS_BOX, n_nodes=cfg.MODEL.FAD.NUM_NODES_BOX) 
            else:        
                # augment 
                bbox_tower = AugmentRCNN(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, cfg.MODEL.FAD.NUM_CHANNELS_BOX, cfg.MODEL.FAD.NUM_CELLS_BOX, cfg.MODEL.FAD.GENO_BOX[0], Cs=cfg.MODEL.FAD.CHANNEL_LIST_BOX, genotypeCH=cfg.MODEL.FAD.GENO_CHANNEL_BOX) 
    
        if cfg.MODEL.FAD.CLSTOWER:
            self.add_module('cls_tower', cls_tower)
        else:
            self.add_module('cls_tower', nn.Sequential(*cls_tower))

        if cfg.MODEL.FAD.BOXTOWER:
                self.add_module('bbox_tower', bbox_tower)
        else:
            self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # ---- 1x1 conv to reduce dim first
        if cfg.MODEL.FAD.CLSTOWER :
            self.cls_reduce = nn.Conv2d(
                in_channels*cfg.MODEL.FAD.NUM_NODES_CLS, in_channels, kernel_size=1, stride=1,
                padding=0
            ) 

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )

        if cfg.MODEL.FAD.BOXTOWER:
            self.box_reduce = nn.Conv2d(
                in_channels*cfg.MODEL.FAD.NUM_NODES_BOX, in_channels, kernel_size=1, stride=1,
                padding=0
            ) 
        
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # Initialization
        list_init = [self.cls_logits, self.bbox_pred]    
    
        if self.cfg.MODEL.FAD.CLSTOWER:            
            list_init.append(self.cls_reduce)    
        else:     
            list_init.append(self.cls_tower)   
        if not self.cfg.MODEL.FAD.BOXTOWER:           
            list_init.append(self.bbox_tower)
        else:
            list_init.append(self.box_reduce)   

        for modules in list_init:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)


    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:

            # ----------- 1st tower 
            if not self.cfg.MODEL.FAD.BOXTOWER:
                bbox_feature = self.bbox_tower(feature)
                bbox_reg.append(self.bbox_pred(bbox_feature))
            else:
                bbox_feature = self.box_reduce(self.bbox_tower(feature))     
                if self.norm_reg_targets:
                    bbox_pred = self.scales[l](self.bbox_pred(bbox_feature))
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
                else:
                    bbox_reg.append(self.bbox_pred(bbox_feature))      

            # ----- 2nd tower
            if not self.cfg.MODEL.FAD.CLSTOWER:
                logits.append(self.cls_logits(self.cls_tower(feature)))
            else:
                cls_tower = self.cls_tower([bbox_feature, feature]) 
                logits.append(self.cls_logits(self.cls_reduce(cls_tower)))

        return logits, bbox_reg


class FADRetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and
    RetinaNet outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FADRetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator_retinanet(cfg)
        head = FADRetinaNetHead(cfg, in_channels)
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        box_selector_test = make_retinanet_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator = make_retinanet_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

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
            box_cls, box_regression = self.head(features, weights_normal)     
        else:       
            box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, box_cls, box_regression, targets):

        loss_box_cls, loss_box_reg = self.loss_evaluator(
            anchors, box_cls, box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_box_cls,
            "loss_retina_reg": loss_box_reg,
        }
        return anchors, losses

    def _forward_test(self, anchors, box_cls, box_regression):
        boxes = self.box_selector_test(anchors, box_cls, box_regression)
        return boxes, {}


def build_fad_retinanet(cfg, in_channels):
    return FADRetinaNetModule(cfg, in_channels)
