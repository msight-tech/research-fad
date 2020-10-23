# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import logging
from .lr_scheduler import WarmupMultiStepLR
import re


def make_optimizer(cfg, model):
    logger = logging.getLogger("fcos_core.trainer")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if cfg.MODEL.FAD.USE_CHANNEL_LR:
            if "box_tower" in key or "cls_tower" in key:
                assert len(cfg.MODEL.FAD.CHANNEL_LIST_CLS) == len(cfg.MODEL.FAD.CHANNEL_LR_CLS)
                assert len(cfg.MODEL.FAD.CHANNEL_LIST_BOX) == len(cfg.MODEL.FAD.CHANNEL_LR_BOX)
                candidate_lrs = cfg.MODEL.FAD.CHANNEL_LR_CLS if "cls_tower" in key \
                    else cfg.MODEL.FAD.CHANNEL_LR_BOX
                info = re.findall(r"dag.(\d).(\d).(\d).(\d)", key)
                assert len(info) == 1 and len(info[0]) == 4
                # get channel index
                channel_idx = int(info[0][2])
                lr = cfg.SOLVER.BASE_LR * candidate_lrs[channel_idx]
            else:
                lr = cfg.SOLVER.BASE_LR
        else:   
            lr = cfg.SOLVER.BASE_LR



        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            logger.info("set lr factor of {} as {}".format(
                key, cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
            ))
            lr *= cfg.SOLVER.DCONV_OFFSETS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
