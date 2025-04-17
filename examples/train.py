import sys

# Import known environments to register them globally
import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.mujoco.cartpole 
# import genRL.gym_envs.test_envs.cartpole_dummy # If needed
from genRL.gym_envs.base import wrap_env_for_training # Import the new wrapper function

import wandb
from genRL.runners import training_loop, get_agent
from genesis.utils.misc import get_platform
import tyro
from genRL.configs import SessionConfig
import gymnasium as gym
import genesis as gs
from genRL.utils import is_cuda_available

# python examples/train.py algo:grpo-config --algo.n_epi 60
# python examples/train.py algo:ppo-config --algo.n_epi 180
# python examples/train.py algo:ppo-config --algo.n_epi 180 --wandb disabled

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
    
    wandb.login()
    run = wandb.init(
                    project=args.project_name,
                    name=f"{args.env_id}-{args.run_name}", # Include env_id in run name
                    config=config,
                    mode=args.wandb,
                    # mode="disabled", # dev dry-run
                )
    
    # --- Environment Creation Logic ---
    # Get environment specification using gym.spec()
    print(f"Getting spec for environment ID: {args.env_id}")
    try:
        env_spec = gym.spec(args.env_id)
    except Exception as e:
        print(f"Error getting spec for environment {args.env_id}: {e}")
        raise e

    # Check the entry point module path from the spec
    if callable(env_spec.entry_point):
        # If entry_point is a class or function, check its module
        entry_point_module = env_spec.entry_point.__module__
    else:
        # If entry_point is a string like "module:callable", extract module part
        entry_point_module = str(env_spec.entry_point).split(":")[0]
        
    is_genesis_env = entry_point_module.startswith("genRL.gym_envs.genesis")
    print(f"Detected entry point module: {entry_point_module}")
    print(f"Is Genesis environment? {is_genesis_env}")

    # Prepare base keyword arguments, removing args handled by wrappers/vectorization
    base_env_kwargs = {
        "seed": args.random_seed,
        "render_mode": "human" if sys.platform == "darwin" else "ansi", # Base render mode
        # Add other args specific to the base envs if needed, filtering out others
        # Example: Pass wandb steps only to Genesis? Might need finer control or envs ignore unknown args
        # "wandb_video_steps": args.wandb_video_steps if args.env_id.startswith("Gen") else None,
    }
    # Add genesis specific args only if it's a genesis env
    if is_genesis_env:
        base_env_kwargs["num_envs"] = config.num_envs
        base_env_kwargs["return_tensor"] = True
        base_env_kwargs["wandb_video_steps"] = args.wandb_video_steps
        base_env_kwargs["logging_level"] = "warning"
        base_env_kwargs["gs_backend"] = gs.gpu if is_cuda_available() else gs.cpu
        
    # Create the (potentially internally vectorized) base environment
    try:
        envs = gym.make(args.env_id, **base_env_kwargs)
        print(f"Base environment {args.env_id} created successfully.")
    except Exception as e:
        print(f"Error creating base environment {args.env_id}: {e}")
        raise e

    # --- Vectorization and Tensor Wrapping Logic ---
    # Determine device based on availability
    device = "cuda" if is_cuda_available() else "cpu"
    
    agent = get_agent(envs, config)
    envs.reset()

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(envs, agent, config, run))
    else:
        training_loop(envs, agent, config, run) # Pass potentially wrapped envs

    # envs.render() # Vector env rendering is handled differently or via wrappers
    envs.close()
    

if __name__ == '__main__':
    main()

