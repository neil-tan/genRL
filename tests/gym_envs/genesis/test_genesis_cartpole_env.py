import pytest
import genRL.gym_envs.genesis.cartpole as gen_cartpole
import genRL.gym_envs.mujoco.cartpole as mujoco_cartpole  # Import to register env
import os
import gymnasium as gym
import torch
import numpy as np

def test_env():
    env = gym.make("GenCartPole-v0",
                   render_mode="ansi",
                   max_force=1000,
                   targetVelocity=5,
                   seed=42)
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    assert obs is not None
    assert reward is not None
    assert done is not None
    assert info is not None
    env.close()

def test_reward_consistency():
    """Test that rewards are consistent and reflect environment dynamics."""
    env = gym.make("GenCartPole-v0",
                  render_mode=None,
                  max_force=1000,
                  targetVelocity=5,
                  seed=42)
    obs, info = env.reset(seed=42)
    
    # Run multiple steps and check if reward is consistent
    rewards = []
    for i in range(50):  # Run for 50 steps
        # Use a consistent action to see predictable behavior
        action = 0.5  # Neutral action (no force)
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
    
    # Test that cart position changes with consistent force
    obs, info = env.reset(seed=42)
    observations = []
    
    # Apply a constant force to move the cart
    for _ in range(50):
        action = 0.75  # Apply force in positive direction
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        
        if terminated or truncated:
            break
    
    # Check that cart position changes over time (first position component)
    if len(observations) > 5:
        initial_pos = observations[0][0]  # First position
        final_pos = observations[-1][0]  # Last position
        assert initial_pos != final_pos, "Cart position should change with applied force"
        # Print for debug purposes
        print(f"Initial position: {initial_pos}, Final position: {final_pos}")
        print(f"Position delta: {final_pos - initial_pos}")
    
    env.close()
    print("Genesis CartPole Reward Consistency Test Passed")

def test_vec_env_step():
    env = gym.make_vec("GenCartPole-v0",
                       num_envs=5,
                       render_mode="ansi",
                       max_force=1000,
                       targetVelocity=5,
                       seed=42)
    env.reset()
    action = torch.tensor([env.action_space.sample()])
    batched_action = torch.stack([action for _ in range(5)])
    
    for i in range(50):
        obs, reward, done, _, info = env.step(batched_action)
        assert obs.shape[0] == 5
        assert reward.shape[0] == 5
        assert reward.dim() == 1
        assert done.shape[0] == 5
        assert done.dim() == 1
        # info not tested

    env.close()

def test_environment_comparison():
    """Test to compare behavior between Genesis and MuJoCo CartPole environments."""
    # Create both environments with same random seed
    seed = 42
    gen_env = gym.make("GenCartPole-v0", render_mode=None, max_force=100, targetVelocity=5, seed=seed)
    mujoco_env = gym.make("MujocoCartPole-v0", render_mode=None, max_force=100, seed=seed)
    
    # Reset both environments
    gen_obs, _ = gen_env.reset(seed=seed)
    mujoco_obs, _ = mujoco_env.reset(seed=seed)
    
    print("Initial observations:")
    print(f"Genesis: {gen_obs}")
    print(f"MuJoCo: {mujoco_obs}")
    
    # Run both environments with the same actions
    gen_positions = []
    mujoco_positions = []
    gen_angles = []
    mujoco_angles = []
    gen_rewards = []
    mujoco_rewards = []
    
    # Use a simple oscillating pattern to test dynamics
    for i in range(100):
        # Oscillating action: vary between 0.25 and 0.75
        action_value = 0.5 + 0.25 * np.sin(i * 0.1)
        gen_action = action_value
        mujoco_action = np.array([action_value * 2 - 1])  # Convert to MuJoCo's [-1, 1] range
        
        # Step both environments
        gen_obs, gen_reward, gen_done, _, _ = gen_env.step(gen_action)
        mujoco_obs, mujoco_reward, mujoco_done, _, _ = mujoco_env.step(mujoco_action)
        
        # Record positions, angles, and rewards
        gen_positions.append(gen_obs[0])
        mujoco_positions.append(mujoco_obs[0])
        gen_angles.append(gen_obs[2])
        mujoco_angles.append(mujoco_obs[2])
        gen_rewards.append(gen_reward)
        mujoco_rewards.append(mujoco_reward)
        
        # Break if either environment terminates
        if gen_done or mujoco_done:
            print(f"Environment terminated after {i+1} steps")
            print(f"Genesis terminated: {gen_done}, MuJoCo terminated: {mujoco_done}")
            break
    
    # Print summary statistics
    print("\nComparison summary:")
    print(f"Genesis steps completed: {len(gen_rewards)}")
    print(f"MuJoCo steps completed: {len(mujoco_rewards)}")
    
    if len(gen_positions) > 0 and len(mujoco_positions) > 0:
        print(f"Genesis position range: [{min(gen_positions):.4f}, {max(gen_positions):.4f}]")
        print(f"MuJoCo position range: [{min(mujoco_positions):.4f}, {max(mujoco_positions):.4f}]")
    
    if len(gen_angles) > 0 and len(mujoco_angles) > 0:
        print(f"Genesis angle range: [{min(gen_angles):.4f}, {max(gen_angles):.4f}]")
        print(f"MuJoCo angle range: [{min(mujoco_angles):.4f}, {max(mujoco_angles):.4f}]")
    
    # Close environments
    gen_env.close()
    mujoco_env.close()

if __name__ == "__main__":
    pytest.main(["-s", "-v", os.path.abspath(__file__)])
