import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import mujoco
# import torch # No longer needed directly here
from genRL.gym_envs.mujoco.base import create_mujoco_single_entry, create_mujoco_vector_entry

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,        # id of the body to track (-1 => world, 0 => cart)
    "distance": 3.0,         # distance from camera to the object
    "lookat": np.array((0.0, 0.0, 3)), # point to look at
    "elevation": -20.0,      # camera rotation around the axis in the xy plane
    "azimuth": 180.0,        # camera rotation around the z axis
}

class MujocoCartPoleEnv(gym.Env):
    """Custom MuJoCo CartPole environment that loads from URDF."""
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "ansi"],
        "render_fps": 100, # Corresponds to model timestep 0.01
    }

    def __init__(
        self,
        frame_skip: int = 1,
        render_mode=None,
        xml_file: str = os.path.join(os.path.dirname(__file__), "../../../../assets/urdf/cartpole.urdf"),
        seed=None,
        camera_config=None,
        max_force = 100.0,
        **kwargs, # Catch unused args
    ):
        print(f"[MujocoCartPoleEnv] Loading model from: {xml_file}")
        self.xml_file = xml_file
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        except Exception as e:
            print(f"[MujocoCartPoleEnv] Error loading model from {self.xml_file}: {e}")
            raise e # Re-raise the error if loading fails
                 
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None

        self.frame_skip = frame_skip
        self.max_force = max_force

        # --- Get joint/DoF indices programmatically --- 
        try:
            # URDF joint names might be different, adjust if needed
            self.slider_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider_to_cart")
            self.hinge_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cart_to_pole")
            
            if self.slider_joint_id == -1:
                 raise ValueError("Joint 'slider_to_cart' not found in the model.")
            if self.hinge_joint_id == -1:
                 raise ValueError("Joint 'cart_to_pole' not found in the model.")

            # Get the DoF index associated with each joint (assuming 1 DoF per joint)
            self.slider_dof_id = self.model.jnt_dofadr[self.slider_joint_id]
            self.hinge_dof_id = self.model.jnt_dofadr[self.hinge_joint_id]
            
            # Get joint ranges
            self.slider_range = self.model.jnt_range[self.slider_joint_id]
            self.hinge_range = self.model.jnt_range[self.hinge_joint_id]
            
        except ValueError as e:
             print(f"[MujocoCartPoleEnv] Error finding joints/DoFs: {e}")
             # You might want to print available joint names here for debugging
             # print("Available joints:", [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)])
             raise e
        # --- End Index Finding --- 

        # Observation space: [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
        # Use determined ranges
        high = np.array(
            [
                self.slider_range[1], # slider range high
                np.inf,              # slider velocity high
                self.hinge_range[1],  # hinge range high
                np.inf,              # hinge velocity high
            ],
            dtype=np.float64,
        )
        low = np.array(
            [
                self.slider_range[0], # slider range low
                -np.inf,             # slider velocity low
                self.hinge_range[0],  # hinge range low
                -np.inf,             # hinge velocity low
            ],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        # Action space: Force applied to the cart (+1 or -1 scaled by max_force)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Termination thresholds (consistent with Gymnasium's standard CartPole)
        self.x_threshold = 2.4 # Use a fixed threshold or derive from slider_range if appropriate
        self.theta_threshold = 12 * 2 * np.pi / 360 # 12 degrees in radians

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._camera_config = camera_config if camera_config is not None else DEFAULT_CAMERA_CONFIG

        self.seed(seed)

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        # Use determined DoF indices
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # Standard gym order: [cart_pos, cart_vel, pole_angle, pole_vel]
        return np.array([position[self.slider_dof_id], velocity[self.slider_dof_id], 
                         position[self.hinge_dof_id], velocity[self.hinge_dof_id]], dtype=np.float64)

    def _apply_action(self, action):
        # Apply force directly to the slider joint's degree of freedom
        force = action[0] * self.max_force
        # Check if qfrc_applied exists and has the expected size
        if hasattr(self.data, 'qfrc_applied') and self.data.qfrc_applied.shape[0] > self.slider_dof_id:
             self.data.qfrc_applied[self.slider_dof_id] = force # Use determined DoF index
        else:
             print(f"[MujocoCartPoleEnv] Warning: qfrc_applied not available or invalid size. Cannot apply force to DoF {self.slider_dof_id}.")

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_obs()
        cart_pos = observation[0] # Corresponds to slider position now
        pole_angle = observation[2] # Corresponds to hinge angle now

        reward = 1.0 # Reward for staying up
        terminated = bool(
            cart_pos < -self.x_threshold
            or cart_pos > self.x_threshold
            or pole_angle < -self.theta_threshold
            or pole_angle > self.theta_threshold
        )
        truncated = False # Not using time limits here, handled by wrappers usually
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbations to initial state using determined DoF indices
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[self.slider_dof_id] = self._np_random.uniform(low=-0.05, high=0.05)
        qpos[self.hinge_dof_id] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[self.slider_dof_id] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[self.hinge_dof_id] = self._np_random.uniform(low=-0.05, high=0.05)
        self.data.qpos = qpos
        self.data.qvel = qvel

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ... _initialize_renderer and render methods remain the same ...
    def _initialize_renderer(self):
        if self.renderer is None:
            if self.render_mode == "human":
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            elif self.render_mode in ["rgb_array", "depth_array"]:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def render(self):
        self._initialize_renderer()
        if self.renderer is None:
            return

        if self.render_mode == "human":
            self.renderer.update_scene(self.data)
            self.renderer.scene.camera.trackbodyid = self._camera_config["trackbodyid"]
            self.renderer.scene.camera.distance = self._camera_config["distance"]
            self.renderer.scene.camera.lookat[:] = self._camera_config["lookat"]
            self.renderer.scene.camera.elevation = self._camera_config["elevation"]
            self.renderer.scene.camera.azimuth = self._camera_config["azimuth"]
            self.renderer.render()
            return None
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "depth_array":
            self.renderer.update_scene(self.data, scene_option=mujoco.MjvOption().flags[mujoco.mjtVisFlag.mjVIS_Depth])
            return self.renderer.render()
        elif self.render_mode == "ansi":
            obs = self._get_obs()
            print(f"Pos:{obs[0]: .2f} Vel:{obs[1]: .2f} Ang:{obs[2]: .2f} AngVel:{obs[3]: .2f}")
            return None

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

# Register the environment using the factory functions from base.py
gym.register(
    id='MujocoCartPole-v0',
    entry_point=create_mujoco_single_entry(MujocoCartPoleEnv), # Pass the class itself
    vector_entry_point=create_mujoco_vector_entry(MujocoCartPoleEnv), # Pass the class itself
    max_episode_steps=1000, 
    reward_threshold=950.0, 
)