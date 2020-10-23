# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from fcos_core.modeling import registry
from fcos_core.modeling.poolers import Pooler
from fcos_core.modeling.make_layers import make_conv3x3


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION
        self.search_head = cfg.MODEL.FAD_ON and cfg.MODEL.FAD.MASKTOWER

        next_feature = input_size
        self.blocks = []

        if not self.search_head:
            for layer_idx, layer_features in enumerate(layers, 1):
                layer_name = "mask_fcn{}".format(layer_idx)
                module = make_conv3x3(
                    next_feature, layer_features,
                    dilation=dilation, stride=1, use_gn=use_gn
                )
                self.add_module(layer_name, module)
                next_feature = layer_features
                self.blocks.append(layer_name)
        else:
            # use nas tower
            from fad_core.modeling.modules.search_rcnn import SearchRCNN
            from fad_core.modeling.modules.augment_rcnn import AugmentRCNN

            if cfg.MODEL.FAD.SEARCH:
                mask_tower = SearchRCNN(next_feature, cfg.MODEL.FAD.NUM_CHANNELS_MASK, cfg.MODEL.FAD.NUM_CELLS_MASK, n_nodes=cfg.MODEL.FAD.NUM_NODES_MASK) 
            else:        
                # augment 
                mask_tower = AugmentRCNN(next_feature, cfg.MODEL.FAD.NUM_CHANNELS_MASK, cfg.MODEL.FAD.NUM_CELLS_MASK, cfg.MODEL.FAD.GENO_MASK[0], Cs=cfg.MODEL.FAD.CHANNEL_LIST_MASK, genotypeCH=cfg.MODEL.FAD.GENO_CHANNEL_MASK) 

            layer_name = "mask_fcn0"
            self.add_module(layer_name, mask_tower)
            self.blocks.append(layer_name)

            # reduce conv
            layer_name = "mask_reduce"
            mask_reduce = nn.Conv2d(next_feature*cfg.MODEL.FAD.NUM_NODES_MASK, next_feature, kernel_size=1, stride=1, padding=0)
            self.add_module(layer_name, mask_reduce)
            self.blocks.append(layer_name)

            layer_features = next_feature # channel remains the same

        self.out_channels = layer_features

    def forward(self, x, proposals, weights_normal=None):
        x = self.pooler(x, proposals)

        #import pdb; pdb.set_trace()
        for layer_name in self.blocks:
            if not self.search_head:
            
                x = F.relu(getattr(self, layer_name)(x))
            else:
                if weights_normal and ('fcn' in layer_name):
                    # searching
                    x = getattr(self, layer_name)(x, weights_normal, [])
                else:
                    # augment
                    x = getattr(self, layer_name)(x)

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
