import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
from genRL.utils import is_cuda_available
import gymnasium as gym
import torch.nn.functional as F
from genRL.runners import get_agent
import genesis as gs
import sys
import wandb
from genRL.runners import training_loop
from genesis.utils.misc import get_platform
import tyro
from genRL.configs import SessionConfig

# python examples/train.py algo:grpo-config --algo.n_epi 65
# python examples/train.py algo:ppo-config --algo.n_epi 185

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole",
                    run_name="cartpole",
                    wandb_video_steps=2000,
                    # algo=PPOConfig(num_envs=8, n_epi=180),
                ),
                description="Minimal RL PPO Cartpole example",
                config=(tyro.conf.ConsolidateSubcommandArgs,),
            )


    # for key, value in dataclasses.asdict(args).items():
    #     print(f"{key}: {value}")
    
    config = args.algo
    
    agent = get_agent(config)
    
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
                   wandb_video_steps=config.wandb_video_steps,
                   logging_level="warning", # "info", "warning", "error", "debug"
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=args.random_seed,
                   )
    
    env.reset()

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(env, agent, config, run))
    else:
        training_loop(env, agent, config, run)

    env.render()
    env.close()
    

if __name__ == '__main__':
    main()

