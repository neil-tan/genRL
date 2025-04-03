import genRL.gym_envs.genesis.cartpole
import gymnasium as gym
import numpy as np
import wandb
import optuna
from genRL.tasks.cartpole import objective
from functools import partial
from genRL.utils import wandb_load_study, wandb_save_study
import tyro
from genRL.configs import PPOConfig, SessionConfig, OptunaConfig
import OpenGL


# run with: python examples/tune_ppo.py --tune.study_name ppo_surr2_fix 
def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole_ppo_tune_gpu",
                    run_name="cartpole",
                    n_epi=10000,
                    wandb_video_steps=2000,
                    tune=OptunaConfig(prune_patience=5, n_trials=200, save_every_n_iters=4),
                ),
                description="Minimal RL PPO Cartpole Hyperparameter tuning example",
            )

    study_name = args.tune.study_name
    save_every_n_iters = args.tune.save_every_n_iters
    n_trials = args.tune.n_trials
    
    if args.fast_dev_run:
        save_every_n_iters = 2
        study_name = f"{study_name}_fast_dev"
        n_trials = 3
    else:
        save_every_n_iters = n_trials if save_every_n_iters == 0 else save_every_n_iters
    
    wandb.login()

    total_trials_completed = 0
    while total_trials_completed < n_trials:
        study, total_trials_completed = wandb_load_study(
                    project_name=args.project_name,
                    study_name=study_name,
                    resume_from_version="latest",
                    direction=args.tune.direction,
                    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(),
                                                        patience=args.tune.prune_patience), # related to report_interval in minimal_rlhf_loop
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
        
        # this will run objective function n_trials times
        study.optimize(partial(objective,
                               run_name=f"{study_name}_{total_trials_completed}",
                               session_config=args,
                                # n_epi=args.n_epi, # this overrides config
                                ),
                n_trials = min(save_every_n_iters, n_trials - total_trials_completed),
                n_jobs = args.tune.n_jobs,
                catch=(OpenGL.error.GLError,),
                )

        total_trials_completed += save_every_n_iters
        
        print("Current Best parameters:", study.best_params)
        print("Current Best value:", study.best_value)
        
        wandb_save_study(study, total_trials_completed, args.project_name, study_name)

if __name__ == '__main__':
    main()