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
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <visual>
    <global offheight="600" offwidth="800"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 .6"/>
    <geom name="ground" type="plane" size=".2 .2 .01" material="grid"
        friction="0.25"/>
    <camera name="closeup" pos="0 -.5 .2" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" friction=".33"/>
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008" friction=".33"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3" friction=".33"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0.01 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# %%
[model.geom(i).name for i in range(model.ngeom)]

mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)
print('raw access:\n', data.geom_xpos)
# %%
# visualize contact frames and forces, make body transparent
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.005
model.vis.scale.forcewidth = 0.01
model.vis.map.force = 0.3
# %%
# enable joint visualization option:
frames = []
duration = 6  # (seconds)
framerate = 60  # (Hz)
dpi = 120
width = 800
height = 600
timevals = []
angular_velocity = []
stem_height = []

# model.opt.gravity = (0, 0, 10)

mujoco.mj_resetDataKeyframe(model, data, 0)
with mujoco.Renderer(model, height=height, width=width) as renderer:
  # mujoco.mj_resetData(model, data)
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, "closeup", options)
      timevals.append(data.time)
      angular_velocity.append(data.qvel[3:6].copy())
      stem_height.append(data.geom_xpos[2,2]);
      pixels = renderer.render()
      frames.append(pixels)
  
media.write_video("mujoco_simulation.mp4", frames, fps=60)
print("Video saved as 'mujoco_simulation.mp4'")
# %%
figsize = (width / dpi, height / dpi)
_, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

ax[0].plot(timevals, angular_velocity)
ax[0].set_title('angular velocity')
ax[0].set_ylabel('radians / second')

ax[1].plot(timevals, stem_height)
ax[1].set_xlabel('time (seconds)')
ax[1].set_ylabel('meters')
_ = ax[1].set_title('stem height')

# %%
