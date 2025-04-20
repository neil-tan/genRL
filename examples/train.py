import sys
import os # Import os

# Set MUJOCO_GL environment variable
os.environ['MUJOCO_GL'] = 'egl'

# Import known environments to register them globally
import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.mujoco.cartpole 

import wandb
from genRL.runners import training_loop, get_agent
from genesis.utils.misc import get_platform
import tyro
from genRL.configs import SessionConfig
import gymnasium as gym
import genesis as gs
from genRL.utils import is_cuda_available

# python examples/train.py --env-id MujocoCartPole-v0 --wandb_video_episodes 30 algo:ppo-config --algo.n-epi 180
# python examples/train.py --env-id MujocoCartPole-v0 --wandb_video_episodes 10 algo:grpo-config --algo.n-epi 60
# python examples/train.py --env-id GenCartPole-v0 --wandb_video_episodes 30 algo:ppo-config --algo.n-epi 180
# python examples/train.py --env-id GenCartPole-v0 --wandb_video_episodes 10 algo:grpo-config --algo.n-epi 60

def main():
    args = tyro.cli(
                SessionConfig,
                default=SessionConfig(
                    project_name="genRL_cartpole",
                    run_name="cartpole_auto_vec", # Generic name
                    env_id="GenCartPole-v0", # Default env
                    wandb_video_episodes=20, # Default to every 20 episodes
                ),
                description="Minimal RL example with auto-vectorization",
            )

    config = args.algo
    
    wandb.login()
    run = wandb.init(
                    project=args.project_name,
                    name=f"{args.env_id}-{args.run_name}", # Include env_id in run name
                    config=config,
                    mode=args.wandb, 
                )
    
    # --- Environment Creation Logic --- 
    device = "cuda" if is_cuda_available() else "cpu"
    gs_backend = gs.gpu if is_cuda_available() else gs.cpu

    # Determine if video recording is needed based on episode count
    video_episodes = getattr(args, 'wandb_video_episodes', None)
    record_video = video_episodes is not None and video_episodes > 0

    # Base envs need rgb_array only if recording video
    base_render_mode = "rgb_array" if record_video else None

    # --- Remove multiprocessing synchronization objects --- 
    # Primitives are now handled internally by the Mujoco factory or not needed for Genesis

    # --- Define kwargs for gym.make_vec --- 
    mjcf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/genRL/gym_envs/mujoco/../../../../assets/mjcf/cartpole_scene.xml"))
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found for training at: {mjcf_path}")
        
    make_vec_kwargs = {
        "env_id": args.env_id,
        "num_envs": config.num_envs,
        "device": device,
        "seed": args.random_seed, 
        "render_mode": base_render_mode, # Workers need rgb_array if recording
        "xml_file": mjcf_path,         # <--- EXPLICITLY ADD MJCF PATH HERE
        "logging_level": "warning",      # Used by Genesis __init__
        "gs_backend": gs_backend,      # Used by Genesis __init__
        "return_tensor": True,         # Used by Genesis __init__
        "max_force": 10.0,             # Now matches Mujoco default
        "targetVelocity": 10,          # Used by Mujoco __init__? Check if needed
        # Pass wandb_video_episodes so the MuJoCo factory can use it
        "wandb_video_episodes": video_episodes,
    }

    print(f"Creating environment {args.env_id} with gym.make_vec.")
    # Use gym.make_vec universally. 
    # For Mujoco, it uses the registered vector_entry_point which handles video wrapping internally.
    # For Genesis, it returns the base env instance (or internally vectorized one).
    envs = gym.make_vec(
        args.env_id, 
        **make_vec_kwargs
    )

    agent = get_agent(envs, config)

    if get_platform() == "macOS" and sys.gettrace() is None:
        gs.tools.run_in_another_thread(fn=training_loop, args=(envs, agent, config, run))
    else:
        training_loop(envs, agent, config, run)

    envs.close()
    
    if run: 
        print("Waiting for wandb run to finish syncing...")
        run.finish()
        print("Wandb run finished.")

if __name__ == '__main__':
    main()
