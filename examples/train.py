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
import tempfile # Re-import tempfile

# --- Remove multiprocessing start method setting ---
# if __name__ == '__main__':
#     try:
#         mp.set_start_method('spawn', force=True)
#         print("[train.py] Set multiprocessing start method to 'spawn'.")
#     except RuntimeError as e:
#         print(f"[train.py] Warning: Could not set start method 'spawn': {e}")

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

    # --- Create temporary directory for video frames (only needed for MuJoCo) ---
    temp_video_dir = None
    video_folder_path = None
    if record_video and "Mujoco" in args.env_id:
        try:
            temp_video_dir = tempfile.TemporaryDirectory()
            video_folder_path = temp_video_dir.name
            print(f"[train.py] Created temporary video folder for MuJoCo: {video_folder_path}")
        except Exception as e:
            print(f"[train.py] Warning: Failed to create temp video directory: {e}. Disabling MuJoCo video recording.")
            record_video = False 
            temp_video_dir = None 
            video_folder_path = None
    # --- End Temp Dir Creation ---

    # Try to get fps from metadata
    try:
        spec = gym.spec(args.env_id)
        fps = spec.metadata.get('render_fps', 30)
    except Exception:
        fps = 30 # Default FPS

    # --- Conditional Environment Creation --- 
    envs = None
    try:
        if args.env_id.startswith("Gen"):
            print(f"Creating Genesis environment {args.env_id} with gym.make.")
            # Create Genesis env directly, passing num_envs and video steps
            envs = gym.make(
                args.env_id,
                num_envs=config.num_envs,
                render_mode="ansi", # Or other modes if needed, Genesis handles rendering
                return_tensor=True, # Genesis should return tensors
                wandb_video_steps=video_steps, # Pass video config directly
                logging_level="warning",
                gs_backend=gs_backend,
                seed=args.random_seed
            )
            # No need for VectorNumpyToTorch for Genesis
            print(f"Genesis environment {args.env_id} created successfully.")

        elif "Mujoco" in args.env_id:
            print(f"Creating MuJoCo environment {args.env_id} with gym.make_vec.")
            # Keyword arguments for the base MuJoCo environment's __init__
            env_kwargs = {
                "seed": args.random_seed, 
                "render_mode": base_render_mode,
            }
            if hasattr(args, 'max_force'): env_kwargs['max_force'] = args.max_force
            
            # Define wrappers list for MuJoCo base envs
            wrappers = []
            if record_video and video_folder_path is not None:
                wrappers.append(partial(
                    RecordSingleEnvVideo,
                    video_folder=video_folder_path, 
                    recorder_lock=recorder_lock,
                    recorder_flag=recorder_flag,
                    name_prefix=f"{args.env_id.replace('-v0','')}-video",
                    video_length=video_steps, 
                    fps=fps
                ))
            
            print(f"Applying base wrappers via gym.make_vec: {wrappers}")
            vec_env = gym.make_vec(
                args.env_id, 
                num_envs=config.num_envs, 
                vectorization_mode="async", 
                wrappers=wrappers if wrappers else None, 
                **env_kwargs
            )
            print(f"Base vectorized MuJoCo environment created successfully.")

            # Apply VectorNumpyToTorch wrapper AFTER vectorization for MuJoCo
            print("Applying VectorNumpyToTorch wrapper.")
            envs = VectorNumpyToTorch(vec_env, device=device)
            print(f"MuJoCo environment {args.env_id} wrapped successfully.")

        else:
            raise ValueError(f"Unsupported environment ID prefix: {args.env_id}")

    except Exception as e:
        print(f"Error creating/wrapping environment {args.env_id}: {e}")
        # Clean up temp dir if creation failed mid-way
        if temp_video_dir:
            try:
                temp_video_dir.cleanup()
                print(f"[train.py] Cleaned up temp video directory after error: {video_folder_path}")
            except Exception as cleanup_e:
                print(f"[train.py] Error cleaning up temp video dir after error: {cleanup_e}")
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
    if temp_video_dir:
         try:
             temp_video_dir.cleanup()
             print(f"[train.py] Cleaned up temp video directory: {video_folder_path}")
         except Exception as e:
             print(f"[train.py] Error cleaning up temp video dir: {e}")

if __name__ == '__main__':
    main()

# TODO:
#     - Review video trigger logic in RecordSingleEnvVideo if needed.
#     - Add more tests for mujoco training