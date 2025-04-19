from dataclasses import dataclass, field
from typing import Union
from typing import Literal

@dataclass
class PPOConfig:
    K_epoch: int = 3
    learning_rate: float = 0.005
    weight_decay: float = 0.00001
    gamma: float = 0.996
    lmbda: float = 0.99
    entropy_coef: float = 0.003
    kl_coef: float = 0.005
    value_loss_coef: float = 1
    normalize_advantage: bool = True
    max_grad_norm: float = 0.15
    eps_clip: float = 0.05
    T_horizon: int = 1500
    num_envs: int = 8
    reward_scale: float = 0.015
    n_epi: int = 1000
    wandb_video_episodes: int = 2000
    report_interval: int = 10
