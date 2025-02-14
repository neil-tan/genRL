# %%
import numpy as np
import genesis as gs
import sys
import gymnasium as gym
from gymnasium import spaces
from transforms3d import euler


# %%
class GenCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}

    def __init__(self, render_mode=None, targetVelocity=0.1, max_force=100, step_scaler:int=1):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.x_threshold = 2.4
        self.theta_threshold_degrees = 12
        self.theta_threshold_radians = self.theta_threshold_degrees * 2 * np.pi / 360 # 0.209
        self.targetVelocity = targetVelocity
        self.max_force = max_force
        self.step_scaler = step_scaler
        self.current_steps_count = 0

        self.done = False

        # [Cart Position, Cart Velocity, Pole Angle (rad), Pole Velocity]
        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -np.inf, -4.1887903e-01, -np.inf]),
                                            np.array([4.8000002e+00, np.inf, 4.1887903e-01, np.inf]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        
        ### simulator setup
        gs.init(backend=gs.cpu)

        self.scene = gs.Scene(
            show_viewer = self.render_mode == "human",
            viewer_options = gs.options.ViewerOptions(
                res           = (2048, 960),
                camera_pos    = (0.0, 8, 0.5),
                camera_lookat = (0.0, 0.0, 3),
                camera_fov    = 60,
                max_FPS       = 60,
            ),
            vis_options = gs.options.VisOptions(
                show_world_frame = True,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = True,
                ambient_light    = (0.1, 0.1, 0.1),
            ),
            renderer=gs.renderers.Rasterizer(),
            rigid_options=gs.options.RigidOptions(
                dt=0.01,
                gravity=(0.0, 0.0, -9.807),
            ),
        )
        
        plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.cartpole = self.scene.add_entity(
            gs.morphs.URDF(file='assets/urdf/cartpole.urdf', fixed=True),
        )
        
        self.cam = self.scene.add_camera(
            res    = (640, 480),
            pos    = (0.0, 8, 0.5),
            lookat = (0.0, 0.0, 3),
            fov    = 60,
            GUI    = False,
        )

        self.scene.build()
        
        self.cartpole.set_dofs_force_range(np.array([-self.max_force]), np.array([self.max_force]), [self.cartpole.get_joint('slider_to_cart').dof_idx_local])
        self._init_state = self.scene.get_state()

        self.reset()

    def _get_obs(self):
        # Returns
        # Cart Position -4.8 ~ 4.8        
        # Cart Velocity -inf ~ inf
        # Pole Angle -0.418 ~ 0.418
        # Pole Velocity At Tip -inf ~ inf
        
        cartpole = self.cartpole

        # assume only moves in x
        cart_position = cartpole.get_link("cart").get_pos()[0]
        cart_velocity = cartpole.get_link("cart").get_vel()[0]
        pole_angular_velocity = cartpole.get_link("pole").get_ang()[1]
        pole_angle = euler.quat2euler(cartpole.get_link("pole").get_quat(), axes='sxyz')[1]
        # pole_position = cartpole.get_link("pole").get_pos()
        
        result = np.array([cart_position, cart_velocity, pole_angle, pole_angular_velocity], dtype=np.float32)
        return result

    def _get_info(self):
        return {}

    def _should_terminate(self, position, angle):
        return (
            position < -self.x_threshold
            or position > self.x_threshold
            or angle < -self.theta_threshold_radians
            or angle > self.theta_threshold_radians
        )

    def _step_simulation(self):
        self.scene.step()
        self.cam.render()
        self.current_steps_count += 1
        # self.scene.viewer.stop()

    def reset(self, seed=None, options=None):
        self._stop_recording()

        super().reset(seed=seed)
        self._seed = seed
        self._options = options
        self.current_steps_count = 0
        self.done = False

        self.scene.reset(self._init_state)

        # randomize initial condition
        random_condition_gen = lambda : np.random.uniform(low=-0.05, high=0.05)
        init_cart_position = random_condition_gen()
        init_cart_velocity = random_condition_gen()
        init_pole_pos = random_condition_gen()
        init_pole_velocity = random_condition_gen()
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.get_joint(name).dof_idx_local for name in jnt_names]
        self.cartpole.set_dofs_position(np.array([init_cart_position, init_pole_pos]), dofs_idx)
        self.cartpole.set_dofs_velocity(np.array([init_cart_velocity, init_pole_velocity]), dofs_idx)

        self.cam.start_recording()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # assert self.action_space.contains(action)
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.get_joint(name).dof_idx_local for name in jnt_names]
        self.cartpole.control_dofs_velocity(np.array([self.targetVelocity if action == 1 else -self.targetVelocity, 0]), dofs_idx)

        self._step_simulation()

        observation = self._get_obs()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        reward = 0
        if not self.done:
            self.done = self._should_terminate(cart_position, pole_angle)
            reward = 1.0

        # observation, reward, done, truncated, info
        return observation, reward, self.done, False, self._get_info()

    
    def getPoleHeight(self):
        pole_height = self.cartpole.get_link("pole").get_AABB()[1,2] \
                        - self.cartpole.get_joint('cart_to_pole').get_pos()[2]

        return pole_height
    
    def getCartPosition(self):
        return self.cartpole.get_link("cart").get_pos()[0]
    
    def render(self):
        if self.render_mode == "human":
            # retina display workaround
            if sys.platform == "darwin" and self.scene._visualizer._viewer is not None:
                self.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
            # This is a blocking call
            # on Mac, it has to run in the main threads
            self.scene.viewer.start()
        elif self.render_mode == "ansi":
            obs = self._get_obs()
            print(f"Cart Position: {obs[0]}; Pole Angle: {obs[2]}")
    
    def close(self):
        self._stop_viewer()
        self._stop_recording()
    
    def _stop_recording(self):
        if self.cam._in_recording:
            self.cam.stop_recording(save_to_filename='video.mp4', fps=60)
    
    def _stop_viewer(self):
        # self.scene._visualizer._viewer
        if self.scene.viewer and self.scene.viewer.is_alive():
            self.scene.viewer.stop()
