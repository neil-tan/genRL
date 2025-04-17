import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.test_envs.cartpole_dummy
import genRL.gym_envs.mujoco.cartpole
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
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

# python examples/train.py algo:grpo-config --algo.n_epi 65
# python examples/train.py algo:ppo-config --algo.n_epi 185

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole",
                    run_name="cartpole_mujoco_ppo_test",
                    wandb_video_steps=2000,
                    # algo=PPOConfig(num_envs=8, n_epi=180),
                ),
                description="Minimal RL PPO Mujoco Cartpole example",
                config=(tyro.conf.ConsolidateSubcommandArgs,),
            )


    # for key, value in dataclasses.asdict(args).items():
    #     print(f"{key}: {value}")
    
    config = args.algo
    
    agent = get_agent(config)
    
    # wandb.login()
    # run = wandb.init(
    #                 project=args.project_name,
    #                 name=args.run_name,
    #                 config=config,
    #                 mode="disabled", # dev dry-run
    #             )
    run = None # Disable wandb run object
    
    # Create vectorized environment
    # Determine device for the agent/buffer
    device = "cuda" if is_cuda_available() else "cpu"
    
    def make_env(seed, render_mode):
        def _init():
            # Pass render_mode for potential video recording, seed for reproducibility
            env = gym.make("MujocoCartPole-v0", seed=seed, render_mode=render_mode)
            # Don't wrap with NumpyToTorch here
            # env = NumpyToTorch(env, device=device)
            return env
        return _init

    # Determine render mode for worker envs (rgb_array needed for recording)
    # For human rendering, typically only one env can render
    worker_render_mode = "rgb_array" # if args.wandb_video_steps else None # Keep rgb_array for now
    
    # Use SyncVectorEnv for simplicity, AsyncVectorEnv for potential speedup
    envs = SyncVectorEnv([
        make_env(args.random_seed + i, worker_render_mode) # Don't pass device
        for i in range(config.num_envs)
    ])
    
    # env = gym.make("GenCartPole-v0",
    # # env = gym.make("GenCartPole-v0-dummy-ones",
    # # env = gym.make("GenCartPole-dummy_inverse_trig-v0",
    #                render_mode="human" if sys.platform == "darwin" else "ansi",
    #                max_force=1000,
    #                targetVelocity=10,
    #                num_envs=config.num_envs,
    #                return_tensor=True,
    #                wandb_video_steps=config.wandb_video_steps,
    #                logging_level="warning", # "info", "warning", "error", "debug"
    #                gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
    #                seed=args.random_seed,
    #                )
    
    # envs.reset() # Reset is typically handled within the training loop

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(envs, agent, config, run))
    else:
        training_loop(envs, agent, config, run) # Pass vectorized envs

    # envs.render() # Vector env rendering is handled differently or via wrappers
    envs.close()
    

if __name__ == '__main__':
    main()

