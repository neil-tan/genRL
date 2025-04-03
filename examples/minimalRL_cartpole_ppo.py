import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
from genRL.utils import is_cuda_available
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from genRL.rl.ppo import PPOConfig
import genesis as gs
import sys
import numpy as np
import wandb
from tqdm import trange
from genRL.tasks.cartpole import training_loop
from genesis.utils.misc import get_platform
import dataclasses
import tyro

@dataclasses.dataclass
class SessionConfig:
    project_name: str
    run_name: str
    n_epi: int
    wandb_video_steps: int
    ppo: PPOConfig

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole_ppo",
                    run_name="cartpole",
                    n_epi=10000,
                    wandb_video_steps=2000,
                    ppo=PPOConfig(normalize_advantage=False, num_envs=8),
                ),
                description="Minimal RL PPO Cartpole example",
            )
    config = args.ppo

    wandb.login()
    run = wandb.init(
                    project=args.project_name,
                    name=args.run_name,
                    config=args.ppo,
                    # mode="disabled", # dev dry-run
                )

    config = PPOConfig(normalize_advantage=False, num_envs=8)

    env = gym.make("GenCartPole-v0",
    # env = gym.make("GenCartPole-v0-dummy-ones",
    # env = gym.make("GenCartPole-dummy_inverse_trig-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config.num_envs,
                   return_tensor=True,
                   wandb_video_steps=config.wandb_video_steps,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=config.random_seed,
                   )
    
    env.reset()
    
    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(env, config, run))
    else:
        training_loop(env, config, run)

    env.render()
    env.close()
    

if __name__ == '__main__':
    main()
