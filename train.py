from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
import yaml
import time
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

"""
def output_dir_modify(file):
    # Load the YAML file
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = config['OUTPUT_DIR']
    # Modify the OUTPUT_DIR parameter
    if 'PATCH_KEEP_RATE' in config['MODEL']:
        if(config['MODEL']['PATCH_KEEP_RATE'] < 1.0):
            pkr = str(config['MODEL']['PATCH_KEEP_RATE'])
            output_dir = output_dir + "_pkr" + pkr
    if 'PFDE_enable' in config['MODEL']:
        if(config['MODEL']['PFDE_enable'] == True):
            output_dir = output_dir + "_pfde"
    if 'FRM_enable' in config['MODEL']:
        if(config['MODEL']['FRM_enable'] == True):
            pkr = str(config['MODEL']['PATCH_KEEP_RATE'])
            output_dir = output_dir + "_frm"
    if 'BDB_enable' in config['MODEL']:
        if(config['MODEL']['PFDE_enable'] == True):
            output_dir = output_dir + "_bdb"
    test_weight = output_dir + "/transformer_120.pth"
    
    config['OUTPUT_DIR'] = output_dir
    config['TEST']['WEIGHT'] = test_weight

    # Save the modified YAML file
    with open(file, 'w') as f:
        yaml.dump(config, f)
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN: # 多卡训练（分布式运算）
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    """
    for param in model.named_parameters():
        print(param[0])
    """
    # print(model)
    
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
