import pytest
import gym_envs.genesis.cartpole as gen_cartpole
import os
import gymnasium as gym
import logging

@pytest.fixture(scope="module", autouse=True)
def target_env():
    custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/gen_cartpole-v1', 
                                                   entry_point=gen_cartpole.GenCartPoleEnv,
                                                   reward_threshold=2000, 
                                                   max_episode_steps=2000,
                                                   )
    env = gym.make(custom_environment_spec, render_mode="ansi", max_force=1000, targetVelocity=5)
    assert env is not None
    yield env
    env.close()

def test_env_step(target_env):
    env = target_env
    env.reset()
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    assert obs is not None
    assert reward is not None
    assert done is not None
    assert info is not None

def test_env_close(target_env):
    env = target_env
    env.close()

if __name__ == "__main__":
    pytest.main(["-s", "-v", os.path.abspath(__file__)])
