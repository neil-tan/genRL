from dataclasses import dataclass, field
from typing import Union
import torch
import numpy as np
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

@dataclass
class OptunaConfig:
    study_name: str = "default_study"
    direction: str = "maximize"
    prune_patience: int = 3
    save_every_n_iters: int = 3
    n_trials: int = 200
    n_jobs: int = 1

@dataclass
class SessionConfig:
    project_name: str
    run_name: str
    # Rename wandb_video_episodes to wandb_video_episodes
    wandb_video_episodes: int = 20
    wandb: Literal["online", "offline", "disabled"] = "online"
    random_seed: int = 42
    fast_dev_run: bool = False
    env_id: str = "GenCartPole-v0"
    algo: PPOConfig | GRPOConfig = field(default_factory=PPOConfig)
    tune: OptunaConfig = field(default_factory=OptunaConfig)
    
    def __post_init__(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
