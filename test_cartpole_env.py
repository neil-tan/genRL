# %%
import numpy as np
import torch
import gymnasium as gym
import custom_envs.cartpole as gen_cartpole
import genesis as gs
import sys

# %%
# set random seed
np.random.seed(98765)
torch.manual_seed(98765)

# %%
custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/gen_cartpole-v1', 
                                                   entry_point=gen_cartpole.GenCartPoleEnv,
                                                   reward_threshold=2000, 
                                                   max_episode_steps=2000,
                                                   )
# %%
env = gym.make(custom_environment_spec, render_mode="human", max_force=1000, targetVelocity=5)

# %%
def training_loop(env, max_steps=300):
    env.reset()
    print_state = lambda obs: print(f"Cart Position: {obs[0]}; Pole Angle: {obs[2]}")
    for i in range(max_steps):
        action = env.action_space.sample()
        # observation, reward, done, truncated, info
        obs, reward, done, _, info = env.step(action)
        if i % 20 == 0:
            print_state(obs)
        if done:
            print(f"Episode finished after {i+1} steps")
            break
    print_state(obs)
    # env.unwrapped.scene.viewer.stop()
    # env.unwrapped._stop_viewer()
    env.close() # stop the viewer and save the video

# %%
if not sys.platform == "linux":
    gs.tools.run_in_another_thread(fn=training_loop, args=(env, 300))
else:
    training_loop(env, 300)

env.render()
# %%
print("finished")