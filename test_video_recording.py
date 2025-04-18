#!/usr/bin/env python
"""
Test script to verify the RecordSingleEnvVideo wrapper works correctly.
This script creates both Genesis and Mujoco environments, applies the wrapper,
and runs them for a few steps to generate videos.
"""
import os
import time
import gymnasium as gym
import multiprocessing as mp
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_video_recording')

# Set MUJOCO_GL environment variable
os.environ['MUJOCO_GL'] = 'egl'

# Import environments
import genRL.gym_envs.genesis.cartpole
import genRL.gym_envs.mujoco.cartpole
from genRL.wrappers.record_single_env_video import RecordSingleEnvVideo

def test_env_video(env_id, num_steps=50, frame_width=None, frame_height=None):
    """Test video recording for a specific environment.
    
    Args:
        env_id: ID of the environment to test
        num_steps: Number of steps to run
        frame_width: Optional fixed frame width
        frame_height: Optional fixed frame height
    """
    logger.info(f"\n--- Testing {env_id} ---")
    logger.info(f"Creating environment with render_mode=rgb_array")
    
    # Create environment with rgb_array render mode
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Apply video wrapper
    video_env = RecordSingleEnvVideo(
        env=env,
        name_prefix=f"test-{env_id}",
        video_length=num_steps,  # Record all steps
        fps=10,
        record_video=True,
        frame_width=frame_width,
        frame_height=frame_height
    )
    
    # Reset the environment
    logger.info("Resetting environment")
    obs, info = video_env.reset()
    
    # Run for specified number of steps
    logger.info(f"Running for {num_steps} steps")
    for i in range(num_steps):
        # Take a random action
        action = video_env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = video_env.step(action)
        
        # Print some information occasionally
        if i % 10 == 0:
            logger.info(f"Step {i+1}/{num_steps}, Reward: {reward:.4f}")
        
        # Reset if episode ended
        if terminated or truncated:
            logger.info("Episode ended, resetting...")
            obs, info = video_env.reset()
    
    # Close the environment to finalize video
    logger.info(f"Closing environment, video should be saved")
    video_env.close()
    logger.info(f"--- Finished testing {env_id} ---\n")

def main():
    """Test video recording for both environment types."""
    logger.info("Starting video recording tests")
    
    # Test with fixed frame dimensions for both environment types
    fixed_width, fixed_height = 320, 240
    
    # Test Genesis environment
    logger.info("Testing Genesis environment with default dimensions")
    test_env_video("GenCartPole-v0")
    
    # Test Genesis environment with fixed dimensions
    logger.info(f"Testing Genesis environment with fixed dimensions {fixed_width}x{fixed_height}")
    test_env_video("GenCartPole-v0", frame_width=fixed_width, frame_height=fixed_height)
    
    # Test Mujoco environment
    logger.info("Testing Mujoco environment with default dimensions")
    test_env_video("MujocoCartPole-v0")
    
    # Test Mujoco environment with fixed dimensions
    logger.info(f"Testing Mujoco environment with fixed dimensions {fixed_width}x{fixed_height}")
    test_env_video("MujocoCartPole-v0", frame_width=fixed_width, frame_height=fixed_height)
    
    logger.info("All tests completed. Check logs for any errors.")

if __name__ == "__main__":
    main()