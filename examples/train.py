import sys
import os # Import os

# Set MUJOCO_GL environment variable
os.environ['MUJOCO_GL'] = 'egl'

# Import known environments to register them globally
import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.mujoco.cartpole 
# from genRL.gym_envs.base import wrap_env_for_training # No longer needed

import wandb
from genRL.runners import training_loop, get_agent
from genesis.utils.misc import get_platform
import tyro
from genRL.configs import SessionConfig
import gymnasium as gym
import genesis as gs
from genRL.utils import is_cuda_available

# python examples/train.py --env-id MujocoCartPole-v0 algo:ppo-config --algo.n-epi 10
# python examples/train.py --env-id GenCartPole-v0 algo:ppo-config --algo.n-epi 10
# python examples/train.py --env-id MujocoCartPole-v0 algo:grpo-config --algo.n-epi 10
# python examples/train.py --env-id GenCartPole-v0 algo:grpo-config --algo.n-epi 10 --wandb offline

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole",
                    run_name="cartpole_auto_vec", # Generic name
                    env_id="GenCartPole-v0", # Default env
                    wandb_video_steps=2000,
                    # Ensure algo default is handled correctly if not provided via cmd line
                ),
                description="Minimal RL example with auto-vectorization",
            )

    # If algo is specified via subcommand (e.g., algo:ppo-config), tyro handles it.
    # If not, it uses the default_factory in SessionConfig.
    config = args.algo
    
    # Restore wandb init - use WANDB_MODE=disabled env var to disable
    wandb.login()
    run = wandb.init(
                    project=args.project_name,
                    name=f"{args.env_id}-{args.run_name}", # Include env_id in run name
                    config=config,
                    mode=args.wandb, # Ensure this uses the attribute from SessionConfig args
                )
    
    # --- Environment Creation Logic --- 
    # Prepare keyword arguments for gym.make_vec
    # These will be passed to the appropriate vector_entry_point
    print(f"Creating vectorized environment {args.env_id} with {config.num_envs} envs.")

    # Use gym.make_vec - it will use the registered vector_entry_point if available
    envs = gym.make_vec(
        args.env_id,
        num_envs=config.num_envs,
        seed=args.random_seed,
        render_mode="human" if sys.platform == "darwin" else "ansi", # Main render mode
        wandb_video_steps=args.wandb_video_steps, # Passed to Genesis
        logging_level="warning", # Passed to Genesis
        gs_backend=gs.gpu if is_cuda_available() else gs.cpu, # Passed to Genesis
        device="cuda" if is_cuda_available() else "cpu", # Passed to Mujoco vector factory for tensor wrapper
        # Add other potential shared args here
    )
    print(f"Vectorized environment {args.env_id} created successfully.")

    # --- End Environment Creation Logic ---
    
    agent = get_agent(envs, config)

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(envs, agent, config, run))
    else:
        training_loop(envs, agent, config, run) # Pass the vectorized envs

    # envs.render() # Vector env rendering is handled differently or via wrappers
    envs.close()
    

if __name__ == '__main__':
    main()

# TODO:
#     - Rendering and wandb video handling for vectorized environments
#     - Add more tests for mujoco training
#     - Mujoco vector or single env wrapper -> base.py -> function factory