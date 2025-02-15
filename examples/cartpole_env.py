# %%
import numpy as np
import torch
import gymnasium as gym
import genRL.gym_envs.genesis.cartpole as gen_cartpole
import genesis as gs
import sys
import time

# %%
# set random seed
np.random.seed(98765)
torch.manual_seed(98765)
# %%
env = gym.make("GenCartPole-v0", render_mode="human", max_force=1000, targetVelocity=5)

# %%
def training_loop(env, max_steps=300):
    env.reset()
    for i in range(max_steps):
        action = env.action_space.sample()
        # observation, reward, done, truncated, info
        obs, reward, done, _, info = env.step(action)
    
        if done:
            break

    env.close() # stop the viewer and save the video

# %%
if not sys.platform == "linux":
    gs.tools.run_in_another_thread(fn=training_loop, args=(env, 300))
else:
    training_loop(env, 300)

# Render every second
while True:
    env.render()
    if env.unwrapped.done == True:
        break
    time.sleep(1)
    
# %%
print("finished")