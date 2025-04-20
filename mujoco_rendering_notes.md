# MujocoCartPoleEnv Rendering Debug Notes

## Goal
Make the `MujocoCartPole-v0` environment rendering look similar to the Genesis environment, including a ground plane, proper lighting, and shadows.

## Attempts

1.  **Camera Adjustment:** Modified `DEFAULT_CAMERA_CONFIG` (distance, lookat) to match Genesis perspective. (Partially successful for camera angle).
2.  **URDF + Visual Settings:** Added `DEFAULT_VISUAL_CONFIG` and applied various `model.vis` settings (ambient, background, quality, map, frame) in `__init__`. Result: No visible change.
3.  **MJCF Conversion Attempt:** Modified URDF to use Mujoco native XML structure. Result: Still no visible change.
4.  **URDF Simplification:** Reverted URDF to near-original + ground plane link. Reverted Python code to minimal visual settings. Result: Still no ground/shadows.
5.  **Debug Script:** Created `debug_render.py`. Result: Confirmed no ground/shadows.
6.  **Programmatic Geom Add (render):** Added plane geom directly in `render()`. Result: Still no ground visible.
7.  **Explicit MjvOption:** Passed `MjvOption` with shadow flag to `update_scene`. Result: Still no ground/shadows.
8.  **MJCF Scene + URDF Include:** Created `cartpole_scene.xml` (ground, light) including `cartpole.urdf`. Result: XML schema errors (`include` children, `material`, `link`).
9.  **MJCF Scene + MJCF Include:** Converted URDF to `cartpole_robot.xml`. Modified scene to include robot MJCF. Result: **Success!** Ground plane and textures visible in `img2txt` render.

## Conclusion
Loading visuals reliably requires using Mujoco's native MJCF format. The best approach is:
1. Define the main scene (static elements like ground, lights) in a parent MJCF file (`cartpole_scene.xml`).
2. Define the robot(s) in separate MJCF files (e.g., by converting from URDF: `cartpole_robot.xml`).
3. Use the `<include>` tag within the parent MJCF to load the robot MJCF(s).
4. Update the Gym environment to load the main scene MJCF file.

## Cleanup
- Revert `MujocoCartPoleEnv` code to load scene MJCF cleanly.
- Delete debug scripts (`debug_render.py`, `convert_urdf.py`) and image (`mujoco_render_test.png`).
- Keep `mujoco_rendering_notes.md`. 