# Other imports and helper functions
# %%
import os
os.environ['MUJOCO_GL'] = 'egl'

import time
import itertools
import numpy as np
import mujoco

import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from IPython.display import clear_output
clear_output()
# %%
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

# %%
data = mujoco.MjData(model)

# %%
# We need to render frames before showing video
renderer = mujoco.Renderer(model, width=320, height=240)
frames = []

# Simulate and render a few frames
for _ in range(100):  # Simulate 100 frames
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

# Now we can show the video
# Save video to a file that can be downloaded/viewed remotely
media.write_video("mujoco_simulation.mp4", frames, fps=60)
print("Video saved as 'mujoco_simulation.mp4'")
# %%
