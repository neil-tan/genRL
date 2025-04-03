from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    K_epoch: int = 5
    learning_rate: float = 0.005
    weight_decay: float = 0.00003
    gamma: float = 0.99
    lmbda: float = 0.998
    entropy_coef: float = 0.001
    value_loss_coef: float = 1
    normalize_advantage: bool = True
    max_grad_norm: float = 0.2
    eps_clip: float = 0.08
    T_horizon: int = 1500
    random_seed: int = 42
    num_envs: int = 32
    reward_scale: float = 0.015
    n_epi: int = 10000
    wandb_video_steps: int = 2000

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
    wandb_video_steps: int
    fast_dev_run: bool = False
    ppo: PPOConfig = field(default_factory=PPOConfig)
    tune: OptunaConfig = field(default_factory=OptunaConfig)