import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
import gymnasium as gym
import torch
import torch.nn.functional as F
from genRL.configs import GRPOConfig
import genesis as gs
import sys
import numpy as np
import wandb
from genRL.runners import training_loop
from genRL.runners import grpo_agent
from genesis.utils.misc import get_platform
import tyro
from genRL.configs import SessionConfig

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole_grpo",
                    run_name="cartpole",
                    wandb_video_episodes=2000,
                    algo=GRPOConfig(num_envs=32, n_epi=180),
                ),
                description="Minimal RL GRPO Cartpole example",
            )
    config = args.algo

    wandb.login()
    run = wandb.init(
                    project=args.project_name,
                    name=args.run_name,
                    config=config,
                    # mode="disabled", # dev dry-run
                )

    env = gym.make("GenCartPole-v0",
    # env = gym.make("GenCartPole-v0-dummy-ones",
    # env = gym.make("GenCartPole-dummy_inverse_trig-v0",
                   render_mode="human" if sys.platform == "darwin" else "ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config.num_envs,
                   return_tensor=True,
                   wandb_video_episodes=config.wandb_video_episodes,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu,
                   seed=config.random_seed,
                   )
    
    env.reset()
    
    agent = grpo_agent(env, config)
    
    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(env, agent, config, run))
    else:
        training_loop(env, agent, config, run)

    env.render()
    env.close()

if __name__ == '__main__':
    main()
