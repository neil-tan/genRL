import pytest
import genRL.gym_envs.genesis.cartpole as gen_cartpole
import os
import gymnasium as gym
import torch

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
        assert done.shape[0] == 5
        # info not tested

    env.close()

if __name__ == "__main__":
    pytest.main(["-s", "-v", os.path.abspath(__file__)])
