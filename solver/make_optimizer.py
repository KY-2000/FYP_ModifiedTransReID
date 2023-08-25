import torch


def make_optimizer(cfg, model, center_criterion):
    params = []

    # 更新learning rate（学习率）和weight decay(权重衰减) [防止过拟合]

    """
    weight decay
    https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
    https://zhuanlan.zhihu.com/p/409346926
    """
    for key, value in model.named_parameters():
        if not value.requires_grad: # if (requires_grad == False)
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR: # fc = fully-connected(全连接层)？
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        """
        momentum explained:
        https://paperswithcode.com/method/sgd-with-momentum
        https://haiping.vip/2020/02/21/%E4%BC%98%E5%8C%96%E5%99%A8/
        """
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY) # 输出新参数？
        """
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    """
    Directly applying Adam optimizer with the hyper-parameters commonly used in ReID community [27]
    to transformer-based models will cause a significant drop in performance.

    AdamW [26] is a commonly used optimizer for training transformer-based models, with much better
    performance compared with Adam. The best results are actually achieved by SGD in our experiments.

    TransReID Thesis, Appendix p.1
    """
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR) # update centers of center loss

    return optimizer, optimizer_center
