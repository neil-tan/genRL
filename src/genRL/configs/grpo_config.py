from dataclasses import dataclass, field
from typing import Union
from typing import Literal

@dataclass
class GRPOConfig:
    K_epoch: int = 6
    learning_rate: float =0.00125
    weight_decay: float = 0.000001
    entropy_coef: float = 0.00015
    kl_coef: float = 0.01
    max_grad_norm: float = 0.2
    eps_clip: float = 0.2
    T_horizon: int = 1500
    num_envs: int = 64
    reward_scale: float = 0.003
    n_epi: int = 1000
    wandb_video_episodes: int = 2000
    report_interval: int = 10