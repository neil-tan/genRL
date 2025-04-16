import pytest
import torch
import numpy as np
import gymnasium as gym
from genRL.runners import get_agent, training_loop
from genRL.configs import PPOConfig, GRPOConfig
import wandb
import genesis as gs
from genRL.utils import is_cuda_available

# Removed custom training loop implementation

def run_training(config):
    """Run training using the shared training_loop and return reward history."""
    # Initialize wandb in offline mode for testing
    wandb.init(mode="disabled")
    
    # Create environment
    env = gym.make("GenCartPole-v0",
                   render_mode="ansi",
                   max_force=1000,
                   targetVelocity=10,
                   num_envs=config.num_envs,
                   return_tensor=True,
                   wandb_video_steps=config.wandb_video_steps,
                   logging_level="warning",
                   gs_backend=gs.gpu if is_cuda_available() else gs.cpu,
                   seed=42)
    
    # Get agent
    agent = get_agent(env, config) # Pass env here
    agent.set_run(wandb.run)  # Set wandb run for logging
    
    # Run training using training_loop
    reward_history = []
    def epi_callback(n_epi, average_score):
        # Move tensor to CPU before appending
        reward_history.append(average_score.cpu())
    
    # Call the shared training_loop with progress bars enabled
    training_loop(env, agent, config, wandb.run, epi_callback=epi_callback)
    
    env.close()
    wandb.finish()
    # Return as a numpy array for consistent snapshotting
    return np.array(reward_history, dtype=np.float32)

# Using a single test with parametrize to cover both algorithms
@pytest.mark.parametrize("algo_config", [
    # n_epi=4, report_interval=4 means callback only at n_epi=4
    PPOConfig(n_epi=4, num_envs=8, report_interval=4),
    GRPOConfig(n_epi=4, num_envs=64, report_interval=4)
])
def test_training_reward_snapshot(snapshot, algo_config):
    """Test that training rewards match the stored snapshot."""
    # Set snapshot directory relative to test file
    snapshot.snapshot_dir = 'tests/snapshots'
    
    reward_history = run_training(algo_config)

    # Check if the history matches the snapshot for the specific config
    # The snapshot filename will include the parametrization details
    # Convert numpy array to bytes for snapshot comparison
    snapshot.assert_match(reward_history.tobytes(), f'{type(algo_config).__name__}_reward_history.npy')
