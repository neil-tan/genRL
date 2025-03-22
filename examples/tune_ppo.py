import genRL.gym_envs.genesis.cartpole
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from genRL.rl.ppo import PPO, SimpleMLP
import genesis as gs
import sys
import numpy as np
import wandb
from tqdm import trange   
import optuna
from genRL.tasks.cartpole import objective
from functools import partial

def main():
    prune_patience = 5
    project_name = "genRL_cartpole_ppo"
    study_name="ppo_cartpole"
    fast_dev_run = True
    save_every_n_iters = 2
    n_trials = 4
    
    if fast_dev_run:
        save_every_n_iters = 2
        study_name = f"{study_name}_fast_dev"
        n_trials = 2
    else:
        save_every_n_iters = n_trials if save_every_n_iters == 0 else save_every_n_iters
    
    wandb.login()
    
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),
                                            patience=prune_patience), # related to report_interval in minimal_rlhf_loop
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    total_trials_completed = 0
    for i in trange(total_trials_completed, n_trials, desc="Optuna Trials"):
        study.optimize(partial(objective,
                                project_name=project_name,
                                fast_dev_run=fast_dev_run),
                n_trials = min(save_every_n_iters, n_trials - i),
                )
        
    print("Current Best parameters:", study.best_params)
    print("Current Best value:", study.best_value)

if __name__ == '__main__':
    main()