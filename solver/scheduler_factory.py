""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS # 最大循环次数
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.SOLVER.BASE_LR # 最低lr
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR # 初始lr
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None

    """
    warmup lr & epoch
    由于刚开始训练时,模型的权重(weights)是随机初始化的，此时若选择一个较大的学习率,可能带来模型的不稳定(振荡)，
    选择Warmup预热学习率的方式，可以使得开始训练的几个epoches或者一些steps内学习率较小,在预热的小学习率下，模
    型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳。
    https://blog.csdn.net/sinat_36618660/article/details/99650804
    https://blog.csdn.net/comway_Li/article/details/105016725
    """

    """
    lr scheduler 学习率调度器
    https://blog.csdn.net/ViatorSun/article/details/123529445
    https://arxiv.org/abs/1608.03983

    - decay rate
    为了防止学习率过大，在收敛到全局最优点的时候会来回摆荡，
    所以要让学习率随着训练轮数不断按指数级下降，收敛梯度下降的学习步长。
    https://blog.csdn.net/qq_40367479/article/details/82530324
    """
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1, #不断缩小lr(new_lr = decay_rate * lr)

            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1, # 最大重启次数
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler
