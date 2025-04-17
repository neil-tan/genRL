import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
import gymnasium as gym
import torch
from genRL.configs import PPOConfig, GRPOConfig
import genesis as gs
import sys
import wandb
from tqdm import trange
import optuna
from dataclasses import replace, asdict
from genRL.rl.agents import ppo_agent, grpo_agent
from typing import Union
from genRL.rl.buffers import SimpleBuffer
from genRL.utils import is_cuda_available

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

def config_grpo_search_space(trial, config:GRPOConfig):
    search_space = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-2),
        "K_epoch": trial.suggest_int("K_epoch", 3, 16),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-4),
        "entropy_coef": trial.suggest_loguniform("entropy_coef", 1e-4, 1e-2),
        "kl_coef": trial.suggest_loguniform("kl_coef", 1e-3, 1e-2),
        "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 0.5),
        "eps_clip": trial.suggest_uniform("eps_clip", 0.05, 0.2),
        "num_envs": trial.suggest_categorical("num_envs", [8, 32, 64]),
        "reward_scale": trial.suggest_loguniform("reward_scale", 0.001, 0.1),
    }
    
    config = replace(config, **search_space)

    return config

def config_search_space(trial, config:Union[PPOConfig, GRPOConfig]):
    """
    Returns the config based on the algo type.
    """
    if isinstance(config, PPOConfig):
        return config_ppo_search_space(trial, config)
    elif isinstance(config, GRPOConfig):
        return config_grpo_search_space(trial, config)
    else:
        raise ValueError(f"Unexpected algo type: {config}")

def training_loop(env, agent, config, run=None, epi_callback=None, compile=False):
    # Determine device based on availability, similar to train.py setup
    device = "cuda" if is_cuda_available() else "cpu"
    model = agent.to(device)
    buffer = SimpleBuffer(config.T_horizon)

    if run is not None:
        wandb.watch([model,], log="all")
    else:
        wandb.init(project="cartpole_ppo", config=config)
        run = wandb.run
    
    model.set_run(run) # for logging

    if compile:
        model.compile()
    
    score = torch.zeros(config.num_envs, device=device)
    report_interval = min(config.report_interval, config.n_epi-1)
    interval_mean_score = None

    epi_bar = trange(config.n_epi, desc="n_epi")
    for n_epi in epi_bar:
        s, _ = env.reset()
        done = False

        with torch.no_grad():
            for t in trange(config.T_horizon, desc="env", leave=False):
                a, log_prob_a, _ = model.pi.sample_action(s)
                s_prime, r, done, truncated, info = env.step(a)

                buffer.add((s, a.detach(), r*config.reward_scale, s_prime, log_prob_a.detach(), done))
                s = s_prime

                score += r

                if done.all():
                    # run.log({"t_end/T_horizon": t/config.T_horizon}) # wandb disabled for now
                    break

        model.train_net(buffer)
        buffer.clear()

        if (n_epi+1)%report_interval==0 or n_epi==0:
            interval_score = (score/report_interval)
            interval_mean_score = (score/report_interval).mean()
            epi_bar.write(f"n_epi: {n_epi+1}, score: {interval_mean_score}")
            run.log({"rewards histo": wandb.Histogram(interval_score.cpu()), "mean reward": interval_mean_score.cpu()})
            if epi_callback is not None:
                epi_callback(n_epi, interval_mean_score)
            score = 0.0
        
    return interval_mean_score


def objective(trial,
              run_name,
              session_config,
              ):

    config = config_search_space(trial, config=session_config.algo)
    
    if session_config.fast_dev_run:
        config = replace(config, n_epi=8, T_horizon=200, wandb_video_steps=20, num_envs=2)

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
                    config=asdict(config),
                    reinit=True,
                    # mode="disabled", # dev dry-run
                )

    env = gym.make("GenCartPole-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config.num_envs,
                   return_tensor=True,
                   wandb_video_steps=config.wandb_video_steps,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=session_config.random_seed,
                   )
    
    env.reset()
    
    agent = get_agent(env, config)

    result = training_loop(env, agent, config, run, epi_callback, compile=True)

    env.render()
    env.close()
    
    return result

def get_agent(env, config: Union[PPOConfig, GRPOConfig]) -> Union[ppo_agent, grpo_agent]:
    """
    Returns the agent based on the config type.
    """
    if isinstance(config, PPOConfig):
        return ppo_agent(env, config)
    elif isinstance(config, GRPOConfig):
        return grpo_agent(env, config)
    else:
        raise ValueError(f"Unexpected algo type: {config}")