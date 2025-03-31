import genRL.gym_envs.genesis.cartpole
import gymnasium as gym
import numpy as np
import wandb
import optuna
from genRL.tasks.cartpole import objective
from functools import partial
from genRL.utils import wandb_load_study, wandb_save_study

def main():
    prune_patience = 3
    project_name = "genRL_cartpole_ppo_tune_gpu"
    study_name="ppo_cartpole"
    fast_dev_run = False
    save_every_n_iters = 3
    n_trials = 200
    n_epi = 800
    
    if fast_dev_run:
        save_every_n_iters = 2
        study_name = f"{study_name}_fast_dev"
        n_trials = 3
    else:
        save_every_n_iters = n_trials if save_every_n_iters == 0 else save_every_n_iters
    
    wandb.login()

    total_trials_completed = 0
    while total_trials_completed < n_trials:
        study, total_trials_completed = wandb_load_study(
                    project_name=project_name,
                    study_name=study_name,
                    resume_from_version="latest",
                    direction="maximize",
                    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),
                                                        patience=prune_patience), # related to report_interval in minimal_rlhf_loop
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
        
        # this will run objective function n_trials times
        study.optimize(partial(objective,
                                project_name=project_name,
                                fast_dev_run=fast_dev_run,
                                n_epi=n_epi, # this overrides config
                                ),
                n_trials = min(save_every_n_iters, n_trials - total_trials_completed),
                )

        total_trials_completed += save_every_n_iters
        
        print("Current Best parameters:", study.best_params)
        print("Current Best value:", study.best_value)
        
        wandb_save_study(study, total_trials_completed, project_name, study_name)

if __name__ == '__main__':
    main()