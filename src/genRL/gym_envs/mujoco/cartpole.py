import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import mujoco
import torch # For potential tensor return types
from genRL.gym_envs.mujoco.base import create_mujoco_single_entry, create_mujoco_vector_entry

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,        # id of the body to track (-1 => world, 0 => cart)
    "distance": 3.0,         # distance from camera to the object
    "lookat": np.array((0.0, 0.0, 0.0)), # point to look at
    "elevation": -20.0,      # camera rotation around the axis in the xy plane
    "azimuth": 180.0,        # camera rotation around the z axis
}

class MujocoCartPoleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "ansi"],
        "render_fps": 100, # Corresponds to model timestep 0.01
    }

    def __init__(
        self,
        frame_skip: int = 1,
        render_mode=None,
        # Change default to point to the URDF file
        xml_file: str = os.path.join(os.path.dirname(__file__), "../../../../assets/urdf/cartpole.urdf"), 
        seed=None,
        camera_config=None,
        max_force = 100.0,
        **kwargs, # Keep kwargs
    ):
        print(f"[MujocoCartPoleEnv] Loading model from: {xml_file}") # Add print to confirm path
        self.xml_file = xml_file
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        except Exception as e:
            print(f"Error loading model from {self.xml_file}: {e}")
            # Attempt to load the original XML as a fallback for debugging
            fallback_xml = os.path.join(os.path.dirname(__file__), "../../../../assets/mujoco/cartpole.xml")
            print(f"Attempting fallback load from: {fallback_xml}")
            try:
                 self.model = mujoco.MjModel.from_xml_path(fallback_xml)
                 print("Fallback XML loaded successfully.")
            except Exception as fallback_e:
                 print(f"Fallback XML load failed: {fallback_e}")
                 raise e # Re-raise the original error
                 
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.renderer = None # Ensure renderer is initialized here

        self.frame_skip = frame_skip
        self.max_force = max_force

        # Observation space: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        high = np.array(
            [
                self.model.jnt_range[0, 1], # slider range
                np.inf,
                self.model.jnt_range[1, 1], # hinge range
                np.inf,
            ],
            dtype=np.float64,
        )
        low = np.array(
            [
                self.model.jnt_range[0, 0], # slider range
                -np.inf,
                self.model.jnt_range[1, 0], # hinge range
                -np.inf,
            ],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        # Action space: Force applied to the cart (+1 or -1 scaled by max_force)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Termination thresholds (consistent with Gymnasium's standard CartPole)
        self.x_threshold = 2.4
        self.theta_threshold = 12 * 2 * np.pi / 360 # 12 degrees in radians

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._camera_config = camera_config if camera_config is not None else DEFAULT_CAMERA_CONFIG

        # --- Remove Video Recording Attributes ---

        self.seed(seed)

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # obs: [cart_pos, pole_angle, cart_vel, pole_vel]
        # Standard gym order: [cart_pos, cart_vel, pole_angle, pole_vel]
        return np.array([position[0], velocity[0], position[1], velocity[1]], dtype=np.float64)

    def _apply_action(self, action):
        # Apply force directly to the slider joint's degree of freedom
        # Assumes the slider joint ('slider_to_cart' from URDF) is the first DoF (index 0)
        force = action[0] * self.max_force
        # Check if qfrc_applied exists and has the expected size
        if hasattr(self.data, 'qfrc_applied') and self.data.qfrc_applied.shape[0] > 0:
             self.data.qfrc_applied[0] = force
        else:
             # This might happen if the URDF parsing leads to unexpected DoF setup
             print("[MujocoCartPoleEnv] Warning: qfrc_applied not available or empty. Cannot apply force.")
        # self.data.ctrl[0] = force # Old method using actuator control

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_obs()
        cart_pos = observation[0]
        pole_angle = observation[2]

        reward = 1.0 # Reward for staying up
        terminated = bool(
            cart_pos < -self.x_threshold
            or cart_pos > self.x_threshold
            or pole_angle < -self.theta_threshold
            or pole_angle > self.theta_threshold
        )
        truncated = False # Not using time limits here, handled by wrappers usually
        info = {}

        if self.render_mode == "human": # Render human mode if not recording rgb
            self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbations to initial state
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[0] = self._np_random.uniform(low=-0.05, high=0.05)
        qpos[1] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[0] = self._np_random.uniform(low=-0.05, high=0.05)
        qvel[1] = self._np_random.uniform(low=-0.05, high=0.05)
        self.data.qpos = qpos
        self.data.qvel = qvel

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _initialize_renderer(self):
        if self.renderer is None:
            if self.render_mode == "human":
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            elif self.render_mode in ["rgb_array", "depth_array"]:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            # else: Do nothing if no render mode requiring a renderer is set

    def render(self):
        self._initialize_renderer()
        if self.renderer is None:
            return # Or raise an error if render called without appropriate mode

        if self.render_mode == "human":
            self.renderer.update_scene(self.data)
            # Setup camera if needed (can be done once or per frame)
            # Example using default config - adjust as needed
            self.renderer.scene.camera.trackbodyid = self._camera_config["trackbodyid"]
            self.renderer.scene.camera.distance = self._camera_config["distance"]
            self.renderer.scene.camera.lookat[:] = self._camera_config["lookat"]
            self.renderer.scene.camera.elevation = self._camera_config["elevation"]
            self.renderer.scene.camera.azimuth = self._camera_config["azimuth"]

            self.renderer.render()
            return None
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            # Return the frame directly for recording
            return self.renderer.render() 
        elif self.render_mode == "depth_array":
            self.renderer.update_scene(self.data, scene_option=mujoco.MjvOption().flags[mujoco.mjtVisFlag.mjVIS_Depth])
            return self.renderer.render()
        elif self.render_mode == "ansi":
            # Simple text rendering for ansi mode
            obs = self._get_obs()
            print(f"Pos:{obs[0]: .2f} Vel:{obs[1]: .2f} Ang:{obs[2]: .2f} AngVel:{obs[3]: .2f}")
            return None # ANSI rendering usually doesn't return data
        # No rendering for ansi mode in MuJoCo

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        
        # No explicit close needed for model/data in mujoco-python > 3.0


# Register the environment using the simplified factory function
gym.register(
    id='MujocoCartPole-v0',
    entry_point=create_mujoco_single_entry(MujocoCartPoleEnv), # Factory for single env (gym.make)
    vector_entry_point=create_mujoco_vector_entry(MujocoCartPoleEnv), # Factory for vectorized env (gym.make_vec)
    max_episode_steps=1000, 
    reward_threshold=950.0, 
)