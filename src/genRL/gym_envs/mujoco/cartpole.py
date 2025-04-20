import os
os.environ['MUJOCO_GL'] = 'egl'
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import mujoco
from functools import partial
from gymnasium.wrappers import NumpyToTorch
from .base import create_mujoco_single_entry, create_mujoco_vector_entry

class MujocoCartPoleEnv(gym.Env):
    """Custom MuJoCo CartPole environment that loads from MJCF scene file."""
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "ansi"],
        "render_fps": 50, # Corresponds to timestep 0.02 in MJCF
    }

    def __init__(
        self,
        frame_skip: int = 1,
        render_mode=None,
        # Default to loading the MJCF scene file
        xml_file: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../assets/mjcf/cartpole_scene.xml")),
        seed=None,
        max_force = 10.0, # Corresponds to standard Gym CartPole control range
        worker_index: int | None = None,
        **kwargs,
    ):
        self.xml_file = xml_file
        self.worker_index = worker_index
        # Store render_mode early for potential use before super().__init__?
        self.render_mode = render_mode

        try:
            # Load the MJCF model
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
            # Physics and visual settings are defined within the MJCF

        except Exception as e:
            print(f"[MujocoCartPoleEnv Worker {self.worker_index}] Error during MJCF model loading: {e}")
            raise e

        self.data = mujoco.MjData(self.model)
        self.renderer = None # Initialize as None

        self.frame_skip = frame_skip
        self.max_force = max_force

        # --- Get joint/DoF indices (names from included model) --- #
        try:
            self.slider_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider_to_cart")
            self.hinge_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_to_pole")
            if self.slider_joint_id == -1: raise ValueError("Joint 'slider_to_cart' not found")
            if self.hinge_joint_id == -1: raise ValueError("Joint 'cart_to_pole' not found")
            self.slider_dof_id = self.model.jnt_dofadr[self.slider_joint_id]
            self.hinge_dof_id = self.model.jnt_dofadr[self.hinge_joint_id]
            self.slider_qpos_adr = self.model.jnt_qposadr[self.slider_joint_id]
            self.hinge_qpos_adr = self.model.jnt_qposadr[self.hinge_joint_id]
            self.slider_qvel_adr = self.model.jnt_dofadr[self.slider_joint_id]
            self.hinge_qvel_adr = self.model.jnt_dofadr[self.hinge_joint_id]
        except ValueError as e:
             print(f"[MujocoCartPoleEnv Worker {self.worker_index}] Error finding joints/DoFs from MJCF: {e}")
             raise e
        # --- End Index Finding --- #

        # --- Observation space (Standard Gym logic) --- #
        self.x_threshold = 2.4 # Standard threshold
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # --- End Observation Space --- #

        # Action space (Standard Gym CartPole)
        self.action_space = spaces.Discrete(2)

        # Ensure render_mode is valid (already stored in self.render_mode)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.seed(seed)

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        # Re-seed the action space for reproducibility if necessary
        # self.action_space.seed(seed)
        return [seed]

    def _get_obs(self):
        # Use determined addresses for state variables
        return np.array([
            self.data.qpos[self.slider_qpos_adr],
            self.data.qvel[self.slider_qvel_adr],
            self.data.qpos[self.hinge_qpos_adr],
            self.data.qvel[self.hinge_qvel_adr]
        ], dtype=np.float32)

    def _apply_action(self, action):
        force = self.max_force if action == 1 else -self.max_force
        if hasattr(self.data, 'qfrc_applied') and self.data.qfrc_applied.shape[0] > self.slider_dof_id:
             self.data.qfrc_applied[self.slider_dof_id] = force
        elif hasattr(self.data, 'ctrl') and self.model.nu > 0:
             actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slider_actuator")
             if actuator_id != -1:
                 self.data.ctrl[actuator_id] = force
             else:
                 # Try first actuator as fallback
                 try: self.data.ctrl[0] = force
                 except IndexError: print("[MujocoCartPoleEnv] Warning: ctrl array empty.")
        else:
             print("[MujocoCartPoleEnv] Warning: Neither qfrc_applied nor ctrl available/applicable.")

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        observation = self._get_obs()
        cart_pos = observation[0]
        pole_angle = observation[2]
        terminated = bool(
            cart_pos < -self.x_threshold
            or cart_pos > self.x_threshold
            or pole_angle < -self.theta_threshold_radians
            or pole_angle > self.theta_threshold_radians
        )
        reward = 1.0 if not terminated else 0.0
        truncated = False
        info = {}
        # Rendering during step is usually handled by wrappers
        # if self.render_mode == "human":
        #     self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.slider_qpos_adr] = self._np_random.uniform(low=-0.05, high=0.05)
        qpos[self.hinge_qpos_adr] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[self.slider_qvel_adr] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[self.hinge_qvel_adr] = self._np_random.uniform(low=-0.05, high=0.05)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()
        info = {}
        return observation, info

    def render(self, mode=None):
        render_mode_to_use = mode if mode is not None else self.render_mode
        # Ensure renderer is initialized; _initialize_renderer handles the None check
        self._initialize_renderer(render_mode_to_use)

        # Handle case where renderer failed to initialize
        if self.renderer is None:
            print("[MujocoCartPoleEnv] Render call skipped: Renderer not initialized.")
            if render_mode_to_use == "ansi": # Still provide ANSI output if possible
                 try: # Wrap in try-except in case _get_obs fails
                     obs = self._get_obs()
                     print(f"Pos:{obs[0]: .2f} Vel:{obs[1]: .2f} Ang:{obs[2]: .2f} AngVel:{obs[3]: .2f}")
                 except Exception as e_ansi:
                     print(f"[MujocoCartPoleEnv] Error getting obs for ANSI render: {e_ansi}")
            # Return a blank frame for rgb_array if renderer is None
            if render_mode_to_use == "rgb_array":
                print("[MujocoCartPoleEnv] Warning: No renderer available, returning blank frame")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            # Return blank depth frame if renderer is None
            if render_mode_to_use == "depth_array":
                 print("[MujocoCartPoleEnv] Warning: No renderer available, returning blank depth frame")
                 return np.zeros((480, 640), dtype=np.float32)
            return None # Return None for other modes (e.g., human)

        # --- Renderer is valid, proceed with rendering --- #
        try:
            # Update the scene with the latest physics state
            self.renderer.update_scene(self.data)
        except Exception as e:
            print(f"[MujocoCartPoleEnv] ERROR updating scene: {e}")
            # If scene update fails, return blank frames for array modes
            if render_mode_to_use == "rgb_array":
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if render_mode_to_use == "depth_array":
                return np.zeros((480, 640), dtype=np.float32)
            return None

        # --- Rendering Logic ---
        if render_mode_to_use == "human":
            try:
                self.renderer.render()
            except Exception as e:
                print(f"[MujocoCartPoleEnv] Error during human rendering: {e}")
            return None # Human rendering doesn't return frames

        elif render_mode_to_use == "rgb_array":
            try:
                frame = self.renderer.render()
                return frame
            except Exception as e:
                print(f"[MujocoCartPoleEnv] Error during rgb_array rendering: {e}")
                # Return blank frame on render error
                return np.zeros((480, 640, 3), dtype=np.uint8)

        elif render_mode_to_use == "depth_array":
            try:
                # Simple fallback - convert RGB to grayscale
                # (Actual depth rendering might need different scene options not used here)
                frame = self.renderer.render()
                if frame is not None:
                    # Ensure it's a numpy array before processing
                    if isinstance(frame, np.ndarray):
                         # Convert RGB to grayscale as a simple depth approximation
                         depth_frame = np.mean(frame, axis=2).astype(np.float32)
                         return depth_frame
                    else:
                         print(f"[MujocoCartPoleEnv] Warning: render() returned unexpected type {type(frame)} for depth fallback")
                         return np.zeros((480, 640), dtype=np.float32)
                else:
                    print("[MujocoCartPoleEnv] Warning: render() returned None during depth fallback")
                    return np.zeros((480, 640), dtype=np.float32)
            except Exception as e:
                print(f"[MujocoCartPoleEnv] Error during depth_array rendering fallback: {e}")
                return np.zeros((480, 640), dtype=np.float32)

        elif render_mode_to_use == "ansi":
            # ANSI already handled if renderer is None
            try:
                obs = self._get_obs()
                print(f"Pos:{obs[0]: .2f} Vel:{obs[1]: .2f} Ang:{obs[2]: .2f} AngVel:{obs[3]: .2f}")
            except Exception as e_ansi:
                 print(f"[MujocoCartPoleEnv] Error getting obs for ANSI render: {e_ansi}")
            return None

        else:
            # Should not happen if render_mode validation is correct
            print(f"[MujocoCartPoleEnv] Warning: Unsupported render mode '{render_mode_to_use}'")
            return None

    def _initialize_renderer(self, render_mode_to_use):
        # Initialize renderer and set camera programmatically
        if self.renderer is None:
            if render_mode_to_use in ["human", "rgb_array", "depth_array"]:
                try:
                    print(f"[MujocoCartPoleEnv Worker {self.worker_index}] Initializing renderer for mode {render_mode_to_use}...")
                    render_width = 640
                    render_height = 480
                    self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
                    print(f"  Renderer instance created: {self.renderer}")
                    print(f"  MuJoCo GL context: {os.environ.get('MUJOCO_GL', 'default')}")

                    # --- Programmatically Set Camera to Track Cart (using mjCAMERA_USER) --- #
                    print("  Setting up camera...")
                    target_body_name = "cart"
                    cart_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
                    print(f"    Target body '{target_body_name}' ID: {cart_body_id}")

                    if cart_body_id != -1:
                        # Access camera settings through the scene
                        # Check model for cameras, not the scene
                        if hasattr(self.renderer, 'scene') and self.model.ncam > 0:
                            camera_id = getattr(self.renderer, 'camera_id', 0) # Default to camera 0
                            print(f"    Attempting to configure camera ID: {camera_id} (model ncam={self.model.ncam})")
                            if camera_id < self.model.ncam:
                                try:
                                    # Apply settings directly to the scene's camera array if possible
                                    if hasattr(self.renderer.scene, 'camera'):
                                        cam = self.renderer.scene.camera[camera_id]
                                        cam.type = mujoco.mjtCamera.mjCAMERA_USER
                                        cam.trackbodyid = cart_body_id
                                        cam.distance = 3.0
                                        cam.elevation = -20
                                        cam.azimuth = 90
                                        print(f"    Camera {camera_id} configured in scene to track body '{target_body_name}'.")
                                    else:
                                        # Fallback: Maybe need to configure camera elsewhere?
                                        # For now, just log that we couldn't set it in the scene.
                                        print("    Warning: renderer.scene does not have 'camera' attribute.")
                                        # We might need to pass camera config to update_scene later instead.
                                except Exception as e:
                                    print(f"    Warning: Failed to set camera properties in scene: {e}")
                            else:
                                 print(f"    Warning: Invalid camera_id {camera_id} for model ncam {self.model.ncam}.")
                        else:
                            print(f"    Warning: Could not access renderer.scene or model has no cameras (ncam={self.model.ncam}).")
                    else:
                        print(f"    Warning: Could not find body '{target_body_name}'. Using default camera.")
                    # --- End Programmatic Camera Setup --- #

                    # --- Lighting Setup --- #
                    # Completely removed any flag setting code here. Rely on defaults.
                    print("  Relying on default MuJoCo lighting.")
                    # --- End Lighting Setup --- #

                    print("  Renderer initialization complete.")

                except Exception as e:
                     # If renderer creation itself fails, print the error
                     print(f"  FATAL: Failed to initialize renderer: {e}")
                     import traceback
                     traceback.print_exc()
                     self.renderer = None # Ensure renderer is None if setup fails
            else:
                # If render mode doesn't require graphical renderer
                print(f"[MujocoCartPoleEnv Worker {self.worker_index}] No graphical renderer needed for mode {render_mode_to_use}.")
        # else:
            # print(f"[MujocoCartPoleEnv Worker {self.worker_index}] Renderer already initialized.")

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        # print(f"[MujocoCartPoleEnv Worker {self.worker_index}] Closed.") # Optional close confirmation

# --- Registration --- #
gym.register(
    id='MujocoCartPole-v0',
    entry_point='genRL.gym_envs.mujoco.cartpole:MujocoCartPoleEnv',
    vector_entry_point=create_mujoco_vector_entry(MujocoCartPoleEnv),
    max_episode_steps=500,
    reward_threshold=475.0,
)