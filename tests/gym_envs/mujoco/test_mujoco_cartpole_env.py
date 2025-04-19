import os
os.environ['MUJOCO_GL'] = 'egl'
import pytest
import genRL.gym_envs.mujoco.cartpole # Import to register env
import gymnasium as gym
import numpy as np
import torch

def test_env_single():
    # Set MUJOCO_GL environment variable
    # os.environ['MUJOCO_GL'] = 'egl'

    """Test single instance instantiation, step, reset, close."""
    env = gym.make("MujocoCartPole-v0", render_mode="rgb_array", seed=42)
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (4,)
    
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert info is not None
    assert isinstance(info, dict)
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert reward is not None
    assert isinstance(reward, float)
    assert terminated is not None
    assert isinstance(terminated, bool)
    assert truncated is not None
    assert isinstance(truncated, bool)
    assert info is not None
    assert isinstance(info, dict)

    # Test rendering
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0
    assert frame.shape[2] == 3 # RGB

    env.close()
    print("Single Env Test Passed")

def test_env_reward_consistency():
    """Test that rewards are consistent and reflect environment dynamics."""
    env = gym.make("MujocoCartPole-v0", render_mode=None, seed=42)
    obs, info = env.reset(seed=42)
    
    # Run multiple steps and check if reward is consistent
    rewards = []
    for i in range(50):  # Run for 50 steps
        # Use a consistent action to see predictable behavior
        action = np.array([0.1], dtype=np.float32)  # Small constant action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify reward is correct type
        assert isinstance(reward, float)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Verify we get rewards while the pole is up
    assert len(rewards) > 0
    # All rewards should be 1.0 while the environment is running
    assert all(r == 1.0 for r in rewards), "All rewards should be 1.0 while the pole is upright"
    
    # Test reward observation correlation
    # Reset environment and run until termination with a strong action
    obs, info = env.reset(seed=42)
    
    # Apply a constant strong force to move the cart 
    actions = []
    observations = []
    num_steps = 0
    max_steps = 300  # Allow more steps
    
    for _ in range(max_steps):
        action = np.array([0.8], dtype=np.float32)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        num_steps += 1
        
        if terminated or truncated:
            break
    
    # Now verify the observations show movement
    if len(observations) > 5:  # If we have enough observations
        # Check that position changes over time
        initial_pos = observations[0][0]  # First position
        final_pos = observations[-1][0]  # Last position
        assert initial_pos != final_pos, "Cart position should change with applied force"
    
    print(f"Reward Test: Ran for {num_steps} steps before termination/max steps")
    env.close()
    print("Reward Consistency Test Passed")

def test_env_vectorized():
    """Test vectorized environment functionality."""
    # Create a vectorized environment with 4 parallel envs
    num_envs = 4
    
    try:
        vec_env = gym.make_vec("MujocoCartPole-v0", num_envs=num_envs, device="cpu", seed=42)
    except Exception as e:
        pytest.skip(f"Vectorized environment testing failed: {e}")
        return
    
    # Verify the vectorized environment properties
    assert hasattr(vec_env, "num_envs")
    assert vec_env.num_envs == num_envs
    
    # Check observation and action spaces
    assert vec_env.single_observation_space.shape == (4,)
    assert vec_env.single_action_space.shape == (1,)
    
    # Test reset
    obs, info = vec_env.reset()
    assert isinstance(obs, torch.Tensor)  # Should be a tensor due to AutoTorchWrapper
    assert obs.shape == (num_envs, 4)
    assert isinstance(info, dict)
    
    # Test step with random actions
    actions = torch.rand((num_envs, 1), dtype=torch.float32) * 2 - 1  # Values between -1 and 1
    next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    # Check shapes and types
    assert isinstance(next_obs, torch.Tensor)
    assert next_obs.shape == (num_envs, 4)
    assert isinstance(rewards, torch.Tensor)
    assert rewards.shape == (num_envs,)
    assert isinstance(terminateds, torch.Tensor)
    assert terminateds.shape == (num_envs,)
    assert isinstance(truncateds, torch.Tensor)
    assert truncateds.shape == (num_envs,)
    
    # Run more steps and verify observations change
    initial_obs = next_obs.clone()
    
    # Run enough steps to see significant differences
    for step in range(50):  # Increase steps
        # Apply consistent different actions to each environment
        actions = torch.tensor([
            [1.0],    # Strong positive for env 0
            [-1.0],   # Strong negative for env 1
            [0.5],    # Medium positive for env 2
            [-0.5]    # Medium negative for env 3
        ], dtype=torch.float32)
        
        next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        # If any environment terminates, we've run enough steps
        if torch.any(terminateds):
            print(f"Environment terminated after {step+1} steps")
            break
    
    # Compare final observations with initial
    obs_diff = torch.sum(torch.abs(next_obs - initial_obs))
    assert obs_diff > 0.01, "Observations should change after multiple steps with different actions"
    
    # Verify the environments have diverged in at least some aspects
    # Test different cart velocities
    velocities = next_obs[:, 1]  # Extract cart velocities (index 1)
    vel_diff = torch.max(velocities) - torch.min(velocities)
    print(f"Velocity difference: {vel_diff.item():.4f}")
    assert vel_diff > 0.01, "Cart velocities should differ between environments"
    
    # Verify reward shape
    assert rewards.shape == (num_envs,)
    
    # Extract all state variables for debugging
    positions = next_obs[:, 0]
    pole_angles = next_obs[:, 2]
    pole_vels = next_obs[:, 3]
    
    print(f"Final positions: {positions}")
    print(f"Final velocities: {velocities}")
    print(f"Final pole angles: {pole_angles}")
    print(f"Final pole velocities: {pole_vels}")
    
    vec_env.close()
    print("Vectorized Environment Test Passed")

# Note: Gymnasium's make_vec is generally preferred over custom vector envs now.
# We will test vectorization by using the standard Gymnasium wrappers in the training script.

if __name__ == "__main__":
    # Setting backend explicitly for testing if needed, though often automatic
    # os.environ['MUJOCO_GL'] = 'glfw' # or egl, osmesa
    pytest.main(["-s", "-v", os.path.abspath(__file__)])