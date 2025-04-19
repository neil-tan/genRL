import pytest
import genRL.gym_envs.mujoco.cartpole # Import to register env
import gymnasium as gym
import numpy as np

def test_env_single():
    # Set MUJOCO_GL environment variable
    # os.environ['MUJOCO_GL'] = 'egl'

    """Test single instance instantiation, step, reset, close."""
    env = gym.make("MujocoCartPole-v0", render_mode="rgb_array", seed=42)
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (4,)
    
    obs, info = env.reset()
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

# Note: Gymnasium's make_vec is generally preferred over custom vector envs now.
# We will test vectorization by using the standard Gymnasium wrappers in the training script.

if __name__ == "__main__":
    # Setting backend explicitly for testing if needed, though often automatic
    # os.environ['MUJOCO_GL'] = 'glfw' # or egl, osmesa
    pytest.main(["-s", "-v", os.path.abspath(__file__)])