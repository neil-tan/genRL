from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    K_epoch: int = 5
    learning_rate: float = 0.005130725958237767
    weight_decay: float = 0.0000412466254419299
    gamma: float = 0.9905071774348913
    lmbda: float = 0.9976561907716108
    entropy_coef: float = 0.0005284384205738843
    value_loss_coef: float = 0.9860861577557356
    normalize_advantage: bool = True
    max_grad_norm: float = 0.2176673586491956
    eps_clip: float = 0.07871516144902298
    T_horizon: int = 1500
    random_seed: int = 42
    num_envs: int = 32
    reward_scale: float = 0.011391114825757769
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