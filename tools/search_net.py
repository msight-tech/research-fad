# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fad_core.engine.trainer import do_train 

from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

from fcos_core.utils.tensorboard import get_tensorboard_writer

from fad_core.modeling.modules.search_rcnn import SearchRCNNController
from fad_core.architect import Architect



def train(cfg, local_rank, distributed, device_ids, use_tensorboard=False):


# ------------------------------- more configs
    half_data = [0,0] # do not split 

    first_order = cfg.SOLVER.SEARCH.FIRST_ORDER    
    alpha_lr = cfg.SOLVER.SEARCH.BASE_LR_ALPHA 
    alpha_weight_decay = 1e-3

    device_ids = [int(x) for x in device_ids]

    if cfg.MODEL.FAD.CLSTOWER or cfg.MODEL.FAD.BOXTOWER:
        n_cells = cfg.MODEL.FAD.NUM_CELLS_CLS
        if cfg.MODEL.FAD.CLSTOWER and cfg.MODEL.FAD.BOXTOWER:
            n_nodes = cfg.MODEL.FAD.NUM_NODES_CLS
            n_module = 2
        elif cfg.MODEL.FAD.CLSTOWER:
            n_nodes = cfg.MODEL.FAD.NUM_NODES_CLS
            n_module = 1
        else:
            n_nodes = cfg.MODEL.FAD.NUM_NODES_BOX
            n_module = 1
    else:
        pdb.set_trace()

    # build model
    model = SearchRCNNController(n_cells, n_nodes=n_nodes, device_ids=device_ids, cfg_det=cfg, n_module=n_module)

    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)
    torch.cuda.set_device(0)
    distributed = False

    if first_order: print('Using 1st order approximationfor the search')

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # ---------------------- optimize alpha
    arch = Architect(model, cfg.SOLVER.MOMENTUM, cfg.SOLVER.WEIGHT_DECAY)
    alpha_optim = torch.optim.Adam(model.alphas(), alpha_lr, betas=(0.5,0.999), weight_decay=alpha_weight_decay)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    # ------ tensorboard
    tb_info = {"tb_logger": None}
    if use_tensorboard:
        tb_logger = get_tensorboard_writer(output_dir)  
        tb_info['tb_logger'] = tb_logger   
        tb_info['prefix'] = cfg.TENSORBOARD.PREFIX


    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        half=half_data[0]
    )

    val_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        half=half_data[1]
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
      
    do_train(
        model,
        arch,
        data_loader,
        val_loader,
        optimizer,
        alpha_optim,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg,
        tb_info=tb_info,
        first_order=first_order,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device_ids", type=list, default=[0])
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(  
        "--use-tensorboard",    
        dest="use_tensorboard",    
        help="Use tensorboardX logger (Requires tensorboardX installed)",         
        action="store_true",  
        default=False    
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # set devices_ids according to num gpus
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    args.device_ids = list(map(str, range(num_gpus)))
    
    # do not use torch.distributed 
    args.distributed = False
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fad_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    model = train(cfg, args.local_rank, args.distributed, args.device_ids, use_tensorboard=args.use_tensorboard)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
