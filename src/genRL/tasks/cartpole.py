import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
from genRL.utils import is_cuda_available
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

#Hyperparameters
# Hyperparameters as a dictionary
config = {
    "learning_rate": 0.0001,
    "weight_decay": 0.000001,
    "gamma": 0.98,
    "lmbda": 0.95,
    "entropy_coef": 0.001,
    "value_loss_coef": 0.5,
    "normalize_advantage": True,
    "max_grad_norm": 0.5,
    "eps_clip": 0.10,
    "T_horizon": 1000,
    "random_seed": 42,
    "num_envs": 16,
    "reward_scale": 0.01,
    "n_epi": 10000,
    "wandb_video_steps": 1500,
}

np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

def get_config(trial, fast_dev_run=False, **kwargs):
    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 5e-5, 1e-2),
        "K_epoch": trial.suggest_categorical("K_epoch", [3, 5, 8, 10, 16]),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-3),
        "gamma": trial.suggest_uniform("gamma", 0.98, 0.999),
        "lmbda": trial.suggest_uniform("lmbda", 0.98, 0.999),
        "entropy_coef": trial.suggest_loguniform("entropy_coef", 1e-6, 1e-3),
        "value_loss_coef": trial.suggest_uniform("value_loss_coef", 0.5, 1.0),
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
        "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 0.5),
        "eps_clip": trial.suggest_uniform("eps_clip", 0.05, 0.2),
        "T_horizon": 1500,
        "random_seed": 42,
        "num_envs": trial.suggest_categorical("num_envs", [1, 8, 32]),
        "reward_scale": trial.suggest_uniform("reward_scale", 0.01, 0.1),
        "n_epi": 15000,
        "wandb_video_steps": 2000,
    }
    
    config.update(kwargs)

    if fast_dev_run:
        config["n_epi"] = 8
        config["wandb_video_steps"] = 20
    return config


def training_loop(env, config, run=None, epi_callback=None, compile=False):
    device = env.unwrapped.device
    
    pi = SimpleMLP(softmax_output=True, input_dim=4, hidden_dim=256, output_dim=2, activation=F.tanh)
    v = SimpleMLP(softmax_output=False, input_dim=4, hidden_dim=256, output_dim=1, activation=F.tanh)

    if run is not None:
        wandb.watch([pi, v], log="all")
    else:
        wandb.init(project="cartpole_ppo", config=config)
        run = wandb.run

    model = PPO(pi=pi, v=v, wandb_run=run, **config).to(device)
    if compile:
        model.compile()
    
    score = torch.zeros(config["num_envs"], device=device)
    report_interval = min(20, config["n_epi"]-1)
    interval_mean_score = None

    epi_bar = trange(config["n_epi"], desc="n_epi")
    for n_epi in epi_bar:
        s, _ = env.reset()
        done = False

        for t in trange(config["T_horizon"], desc="env", leave=False):
            prob = model.pi(s)
            m = Categorical(prob)
            a = m.sample().unsqueeze(-1)
            s_prime, r, done, truncated, info = env.step(a)

            prob_a = torch.gather(prob, -1, a)
            model.put_data((s, a.detach(), r*config["reward_scale"], s_prime, prob_a.detach(), done))
            s = s_prime

            score += r
            
            done = done.all() if isinstance(done, torch.Tensor) else done
            if done:
                run.log({"t_end/T_horizon": t/config["T_horizon"]})
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
              project_name,
              run_name="cartpole",
              fast_dev_run=False,
              **kwargs):

    config = get_config(trial, fast_dev_run=fast_dev_run, **kwargs)

    def epi_callback(n_epi, average_score):
        trial.report(average_score, n_epi)
        if trial.should_prune():
            print("\033[93mPruning trial\033[0m")
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()
    
    run = wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                    reinit=True,
                    # mode="disabled", # dev dry-run
                )

    env = gym.make("GenCartPole-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config["num_envs"],
                   return_tensor=True,
                   wandb_video_steps=config["wandb_video_steps"],
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=config["random_seed"],
                   )
    
    env.reset()
    
    result = training_loop(env, config, run, epi_callback, compile=True)

    env.render()
    env.close()
    
    return result