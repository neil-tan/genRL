# tests/gym_envs/mujoco/test_mujoco_cartpole_training.py
import pytest
import torch
import numpy as np
import gymnasium as gym
import genRL.gym_envs.mujoco.cartpole # Import to register the environment
from genRL.runners import get_agent, training_loop
from genRL.configs import PPOConfig # Assuming PPO is the primary algo for Mujoco tests
import wandb
import os
from genRL.utils import is_cuda_available

# Set MUJOCO_GL environment variable for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

def run_training(config):
    """Run training using the shared training_loop for MujocoCartPole-v0."""
    # --- Add settings for determinism --- 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # --- End settings for determinism ---
    
    # Initialize wandb in offline mode for testing
    wandb.init(mode="disabled")

    # Determine device
    device = "cuda" if is_cuda_available() else "cpu"

    # Prepare kwargs for gym.make_vec
    vec_env_kwargs = {
        "seed": 42,
        "render_mode": "rgb_array", # Use non-human mode for testing
        "device": device, # Pass device for the wrapper
        # Add other relevant kwargs if needed, filtering out Genesis-specific ones
    }

    # Create vectorized environment using gym.make_vec and vector_entry_point
    try:
        env = gym.make_vec(
            "MujocoCartPole-v0",
            num_envs=config.num_envs,
            **vec_env_kwargs
        )
        print(f"MujocoCartPole-v0 vectorized env created successfully.")
    except Exception as e:
        print(f"Error creating MujocoCartPole-v0 vec env: {e}")
        # Check spec for debugging
        try:
            spec = gym.spec("MujocoCartPole-v0")
            print(f"Env Spec: {spec}")
            print(f"Vector entry point: {getattr(spec, 'vector_entry_point', 'Not defined')}")
        except Exception as spec_e:
            print(f"Could not get spec: {spec_e}")
        raise e

    # Get agent (should handle Mujoco's continuous action space correctly now)
    agent = get_agent(env, config)
    agent.set_run(wandb.run)  # Set wandb run for logging

    # Run training using training_loop
    reward_history = []
    def epi_callback(n_epi, average_score):
        # Move tensor to CPU before appending
        if isinstance(average_score, torch.Tensor):
             reward_history.append(average_score.cpu().numpy()) # Convert tensor to numpy
        else:
             reward_history.append(np.array(average_score)) # Ensure it's numpy

    # Call the shared training_loop
    training_loop(env, agent, config, wandb.run, epi_callback=epi_callback)

    env.close()
    wandb.finish()
    # Return as a numpy array for consistent snapshotting
    # Ensure all elements are numpy arrays before stacking
    return np.array(reward_history, dtype=np.float32)

# Using PPO for Mujoco testing
@pytest.mark.parametrize("algo_config", [
    PPOConfig(n_epi=10, num_envs=8, report_interval=5, T_horizon=100), # Short horizon for faster test
])
def test_training_reward_snapshot(snapshot, algo_config):
    """Test that training rewards match the stored snapshot for Mujoco."""
    # Set snapshot directory relative to test file
    snapshot.snapshot_dir = 'tests/snapshots'

    reward_history = run_training(algo_config)

    # Check if the history matches the snapshot for the specific config
    snapshot_filename = f'Mujoco_{type(algo_config).__name__}_reward_history.npy'
    snapshot.assert_match(reward_history.tobytes(), snapshot_filename)
