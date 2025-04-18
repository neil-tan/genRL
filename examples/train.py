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
from genRL.wrappers.vector_numpy_to_torch import VectorNumpyToTorch # Import tensor wrapper
from genRL.wrappers.record_single_env_video import RecordSingleEnvVideo # Import new wrapper
from functools import partial # Import partial for wrapper factory
import multiprocessing as mp # Import multiprocessing

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
    device = "cuda" if is_cuda_available() else "cpu"
    gs_backend = gs.gpu if is_cuda_available() else gs.cpu

    # Determine if video recording is needed
    video_steps = getattr(config, 'wandb_video_steps', None)
    record_video = video_steps is not None and video_steps > 0
    
    # Base envs need rgb_array only if recording video for Mujoco
    base_render_mode = "rgb_array" if record_video and "Mujoco" in args.env_id else None 

    # --- Create multiprocessing synchronization objects (only needed for MuJoCo) ---
    recorder_lock = mp.Lock() if "Mujoco" in args.env_id else None
    recorder_flag = mp.Value('i', 0) if "Mujoco" in args.env_id else None

    # --- Remove temporary directory creation --- 

    # Try to get fps from metadata
    try:
        spec = gym.spec(args.env_id)
        fps = spec.metadata.get('render_fps', 30)
    except Exception:
        fps = 30 # Default FPS

    # --- Define kwargs for gym.make_vec --- 
    # These will be passed to the vector_entry_point (for MuJoCo) 
    # or potentially the base env __init__ (for Genesis)
    make_vec_kwargs = {
        "env_id": args.env_id,         # Pass env_id explicitly
        "num_envs": config.num_envs,
        "seed": args.random_seed, 
        "render_mode": base_render_mode, # Used by MuJoCo vector entry point
        "logging_level": "warning",      # Used by Genesis __init__
        "gs_backend": gs_backend,      # Used by Genesis __init__
        "return_tensor": True,         # Used by Genesis __init__
        "wandb_video_steps": video_steps, # Used by Genesis __init__
        # Args for MuJoCo video recording (passed to its vector_entry_point)
        "record_video": record_video, 
        "recorder_lock": recorder_lock,
        "recorder_flag": recorder_flag,
        "name_prefix": f"{args.env_id.replace('-v0','')}-video",
        "video_length": video_steps if video_steps else 1000,
        "fps": fps,
    }
    # Add env-specific kwargs from args
    if hasattr(args, 'max_force'): make_vec_kwargs['max_force'] = args.max_force
    if hasattr(args, 'targetVelocity'): make_vec_kwargs['targetVelocity'] = args.targetVelocity

    print(f"Creating environment {args.env_id} with gym.make_vec (using vector_entry_point if available).")
    try:
        # Use gym.make_vec universally. It will use the registered vector_entry_point if found.
        # For Genesis, it should ideally fall back to calling gym.make with num_envs.
        # We don't specify vectorization_mode, letting make_vec choose based on entry point.
        vec_env = gym.make_vec(
            args.env_id, 
            # num_envs is now in kwargs
            # wrappers=None, # Wrappers are handled by vector_entry_point or not needed
            **make_vec_kwargs
        )
        print(f"Environment {args.env_id} created successfully.")

        # --- Apply Post-Vectorization Wrappers --- 
        # Apply Tensor Wrapper conditionally AFTER vectorization (if needed)
        if "Mujoco" in args.env_id:
            print("Applying VectorNumpyToTorch wrapper.")
            envs = VectorNumpyToTorch(vec_env, device=device)
        else:
            envs = vec_env # Genesis env already returns tensors
            
        print(f"Environment {args.env_id} final setup complete.")

    except Exception as e:
        print(f"Error creating/wrapping environment {args.env_id}: {e}")
        # Clean up temp dir if creation failed mid-way
        raise e

    # --- End Environment Creation Logic ---
    
    # Ensure envs is not None before proceeding
    if envs is None:
        raise RuntimeError("Environment creation failed.")
        
    agent = get_agent(envs, config)

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(envs, agent, config, run))
    else:
        training_loop(envs, agent, config, run) # Pass the vectorized envs

    # envs.render() # Vector env rendering is handled differently or via wrappers
    envs.close()
    # --- Clean up temporary directory (only if created) --- 

if __name__ == '__main__':
    main()

# TODO:
#     - Review video trigger logic in RecordSingleEnvVideo if needed.
#     - Add more tests for mujoco training