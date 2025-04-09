import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
from genRL.utils import is_cuda_available
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from genRL.rl.ppo import PPO, SimpleMLP, PPOConfig
import genesis as gs
import sys
import numpy as np
import wandb
from tqdm import trange
import optuna
from dataclasses import replace, asdict

np.random.seed(PPOConfig.random_seed)
torch.manual_seed(PPOConfig.random_seed)

def config_ppo_search_space(trial, config:PPOConfig):
    search_space = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-2),
        "K_epoch": trial.suggest_categorical("K_epoch", [3, 5, 8, 16]),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-4),
        "gamma": trial.suggest_uniform("gamma", 0.985, 0.999),
        "lmbda": trial.suggest_uniform("lmbda", 0.985, 0.999),
        "entropy_coef": trial.suggest_loguniform("entropy_coef", 1e-4, 1e-2),
        "kl_coef": trial.suggest_loguniform("kl_coef", 1e-3, 1e-2),
        "value_loss_coef": trial.suggest_uniform("value_loss_coef", 0.8, 1.0),
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
        "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 0.5),
        "eps_clip": trial.suggest_uniform("eps_clip", 0.05, 0.15),
        "num_envs": trial.suggest_categorical("num_envs", [1, 8, 32, 64]),
        "reward_scale": trial.suggest_uniform("reward_scale", 0.01, 0.05),
    }
    
    config = replace(config, **search_space)

    return config


def training_loop(env, config, run=None, epi_callback=None, compile=False):
    device = env.unwrapped.device
    
    pi = SimpleMLP(softmax_output=True, input_dim=4, hidden_dim=256, output_dim=2)
    v = SimpleMLP(softmax_output=False, input_dim=4, hidden_dim=256, output_dim=1)

    if run is not None:
        wandb.watch([pi, v], log="all")
    else:
        wandb.init(project="cartpole_ppo", config=config)
        run = wandb.run

    model = PPO(pi=pi, v=v, wandb_run=run, config=config).to(device)
    if compile:
        model.compile()
    
    score = torch.zeros(config.num_envs, device=device)
    report_interval = min(20, config.n_epi-1)
    interval_mean_score = None

    epi_bar = trange(config.n_epi, desc="n_epi")
    for n_epi in epi_bar:
        s, _ = env.reset()
        done = False

        with torch.no_grad():
            for t in trange(config.T_horizon, desc="env", leave=False):
                prob = model.pi(s)
                m = Categorical(prob)
                a = m.sample().unsqueeze(-1)
                s_prime, r, done, truncated, info = env.step(a)

                prob_a = torch.gather(prob, -1, a)
                model.put_data((s, a.detach(), r*config.reward_scale, s_prime, prob_a.detach(), done))
                s = s_prime

                score += r
                
                done = done.all() if isinstance(done, torch.Tensor) else done
                if done:
                    run.log({"t_end/T_horizon": t/config.T_horizon})
                    break

        model.train_net()

        if n_epi%report_interval==0 and n_epi!=0:
            interval_score = (score/report_interval)
            interval_mean_score = (score/report_interval).mean()
            epi_bar.write(f"n_epi: {n_epi}, score: {interval_mean_score}")
            run.log({"rewards histo": wandb.Histogram(interval_score.cpu()), "mean reward": interval_mean_score.cpu()})
            if epi_callback is not None:
                epi_callback(n_epi, interval_mean_score)
            score = 0.0
        
    return interval_mean_score

def objective(trial,
              run_name,
              session_config,
              ):

    ppo_config = config_ppo_search_space(trial, config=session_config.ppo)
    
    if session_config.fast_dev_run:
        ppo_config = replace(ppo_config, n_epi=8, T_horizon=200, wandb_video_steps=20, num_envs=2)

    def epi_callback(n_epi, average_score):
        trial.report(average_score, n_epi)
        if trial.should_prune():
            print("\033[93mPruning trial\033[0m")
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()
    
    run = wandb.init(
                    project=session_config.project_name,
                    name=run_name,
                    config=asdict(ppo_config),
                    reinit=True,
                    # mode="disabled", # dev dry-run
                )

    env = gym.make("GenCartPole-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=ppo_config.num_envs,
                   return_tensor=True,
                   wandb_video_steps=ppo_config.wandb_video_steps,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=ppo_config.random_seed,
                   )
    
    env.reset()
    
    result = training_loop(env, ppo_config, run, epi_callback, compile=True)

    env.render()
    env.close()
    
    return result