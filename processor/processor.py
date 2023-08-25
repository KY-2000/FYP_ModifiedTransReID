import logging
import os
import time
import datetime
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import GPUtil
from tensorboardX import SummaryWriter
# from visualize.grad_cam_transreid import Visualization
import cv2
import numpy as np


def get_GPU_status():
    used_gpus = [gpu for gpu in GPUtil.getGPUs() if str(gpu.id) in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    log_arr = []
    for gpu in used_gpus:
        log = "GPU " + str(gpu.id) + ": Memory Used " + str(gpu.memoryUsed) + "MB / Memory Total " + str(gpu.memoryTotal) + "MB"
        log_arr.append(log)
    return log_arr

def do_train(cfg,
             model,
             center_criterion, # center loss
             train_loader, # training dataset
             val_loader, # validation dataset
             optimizer, # parameter update mechanism
             optimizer_center, # updated centers of center loss
             scheduler, # lr scheduler
             loss_fn, # loss function(ID loss, tripltes loss or cross entropy loss)
             num_query, # number of data in dataset(training + validation)
             local_rank): # 分布式运算相关（LOCAL_RANK：当前进程对应的GPU号）
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device: # 多卡训练
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    logger.info(model)
    loss_meter = AverageMeter() # loss
    acc_meter = AverageMeter() # accuracy

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM) # 输出R1和mAP（实验数据）
    scaler = amp.GradScaler() # 梯度调节（加速训练）
    # train
    for epoch in range(1, epochs + 1):
        if(epoch == 1): 
            training_start = time.time()
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch) # scheduler.py
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad() # 梯度置零
            optimizer_center.zero_grad() # 梯度置零
            img = img.to(device) # 输入
            target = vid.to(device) # 目标输出
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                # 加速训练，https://blog.csdn.net/weixin_42216799/article/details/110876374
                # https://zhuanlan.zhihu.com/p/348554267
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward() # calc gradient
            """
            https://zhuanlan.zhihu.com/p/83172023
            """

            scaler.step(optimizer)
            scaler.update() # update scale factor

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                """
                for info in gpu_info:
                    logger.info(info)
                """

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        log_arr = get_GPU_status()
        for log in log_arr:
            logger.info(log)
        
        if cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done.".format(epoch))
            # pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            logger.info('Done training')
            training_end = time.time()
            training_period = str(datetime.timedelta(seconds = (training_end - training_start)))
            logger.info("Total Training Time: {}".format(training_period))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                    # saves and loads only the model parameters
                    # https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            val_start = time.time()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    """
                    train() v eval()
                    https://www.cnblogs.com/luckyplj/p/13424561.html

                    model.eval() 作用等同于 self.train(False)。简而言之，就是评估模式。而非训练模式。
                    在评估模式下, batchNorm层, dropout层等用于优化训练而添加的网络层会被关闭, 从而使得评估时不会发生偏移。
                    https://blog.csdn.net/weixin_43977640/article/details/109694244

                    no_grad()
                    set the attribute required_grad of tensor False and deactivates the Autograd engine
                    """
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad(): # no_grad()与eval()配合使用
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    val_end = time.time()
                    val_period = str(datetime.timedelta(seconds = (val_end - val_start)))
                    logger.info("Total Validation Time: {}".format(val_period))
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                        if n_iter == 0:
                            weights_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
                            Visualization(weights_path, imgpath)

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                val_end = time.time()
                val_period = str(datetime.timedelta(seconds = (val_end - val_start)))
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            #print(img[0].shape) # [3,256,128]
            #print(imgpath)
            #img = np.array(img[0].cpu()).transpose(1, 2, 0)
            #cv2.imwrite("test2.jpg", img) 
            img = img.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            """
            if n_iter == 0:
                Visualization(model, img, camids, target_view)
            """
            
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


