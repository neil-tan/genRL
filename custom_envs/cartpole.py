# %%
import numpy as np
import genesis as gs
import sys
import gymnasium as gym
from gymnasium import spaces


# %%
class CartPolePyBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, targetVelocity=0.1, max_force=100, step_scaler:int=1):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.x_threshold = 2.4
        self.theta_threshold_degrees = 12
        self.theta_threshold_radians = self.theta_threshold_degrees * 2 * np.pi / 360 # 0.209
        self.targetVelocity = targetVelocity
        self.max_force = max_force
        self.step_scaler = step_scaler

        self.done = False

        # [Cart Position, Cart Velocity, Pole Angle (rad), Pole Velocity]
        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
                                            np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        
        ### simulator setup
        gs.init(backend=gs.cpu)

        self.scene = gs.Scene(
            show_viewer = True,
            viewer_options = gs.options.ViewerOptions(
                # res           = (1280, 960),
                camera_pos    = (3.5, 0.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
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
            pos    = (3.5, 0.0, 2.5),
            lookat = (0, 0, 0.5),
            fov    = 45,
            GUI    = False,
        )

        self.scene.build()
        
        self.cartpole.set_dofs_force_range(-self.max_force, self.max_force)
        self._init_state = self.get_state()
        
        # if not sys.platform == "linux":
        #     if sys.platform == "darwin" and scene._visualizer._viewer is not None:
        #         scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
        #     gs.tools.run_in_another_thread(fn=run_sim, args=(scene, cartpole, cam))
        # else:
        #     run_sim(scene, cartpole, cam)
        # scene.viewer.start()

        # self.physID = p.connect(p.DIRECT)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def _get_obs(self):
        # Returns
        # Cart Position -4.8 ~ 4.8        
        # Cart Velocity -inf ~ inf
        # Pole Angle -0.418 ~ 0.418
        # Pole Velocity At Tip -inf ~ inf
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.cartpole.get_joint(name).dof_idx_local for name in jnt_names]

        # assume only moves in x
        cart_position = self.cartpole.get_dofs_position(dofs_idx)[0]
        # cart_position = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)[0][0]
        cart_velocity = self.cartpole.get_dofs_velocity(dofs_idx)[0]
        # cart_velocity = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)[6][0]
        
        _, pole_angular_velocity, pole_angle = self._getPoleStates(self.cartpole)
        
        result = np.array([cart_position, cart_velocity, pole_angle, pole_angular_velocity], dtype=np.float32)
        return result

    def _get_info(self):
        return {}
        
    def _getPoleStates(self, cartpole):
        link_state = p.getLinkState(cartpole, 1, computeLinkVelocity=1, physicsClientId=self.physID)
        position = link_state[0]
        angular_velocity = link_state[7][1]
        # assuming the pole is not rotating around the x and y axis
        angle = p.getAxisAngleFromQuaternion(link_state[5], physicsClientId=self.physID)[1]
        return position, angular_velocity, angle

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
        # self.scene.viewer.stop()

    def reset(self, seed=None, options=None):
        self.cam.stop_recording(save_to_filename='video.mp4', fps=60)
        super().reset(seed=seed)
        self._seed = seed
        self._options = options
        self.current_steps_count = 0
        self.done = False

        self.scene.reset(self._init_state)
        # p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=self.physID)

        # randomize initial condition
        random_condition_gen = lambda : np.random.uniform(low=-0.05, high=0.05)
        init_cart_position = random_condition_gen()
        init_cart_velocity = random_condition_gen()
        init_pole_pos = random_condition_gen()
        init_pole_velocity = random_condition_gen()
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.cartpole.get_joint(name).dof_idx_local for name in jnt_names]
        self.cartpole.cartpole.set_dofs_position(np.array([init_cart_position, init_pole_pos]), dofs_idx)
        self.cartpole.cartpole.set_dofs_velocity(np.array([init_cart_velocity, init_pole_velocity]), dofs_idx)

        self.cam.start_recording()
        self._step_simulation()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # position, orientation = p.getBasePositionAndOrientation(self.cartpole)
        # x, y, z = position
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.cartpole.get_joint(name).dof_idx_local for name in jnt_names]
        self.cartpole.control_dofs_velocity(np.array([self.targetVelocity if action == 1 else -self.targetVelocity, 0]), dofs_idx)

        # p.setJointMotorControl2(self.cartpole, 0,
        #                         p.VELOCITY_CONTROL,
        #                         targetVelocity=self.targetVelocity if action == 1 else -self.targetVelocity,
        #                         force=self.max_force,
        #                         physicsClientId=self.physID)

        self._step_simulation()

        observation = self._get_obs()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        reward = 0
        if not self.done:
            self.done = self._should_terminate(cart_position, pole_angle)
            reward = 1.0

        return observation, reward, self.done, False, self._get_info()

    
    def getPoleHeight(self):
        pole_aabb_max = p.getAABB(self.cartpole, 1, physicsClientId=self.physID)[1]   
        cart_aabb_max = p.getAABB(self.cartpole, 0, physicsClientId=self.physID)[1]

        return pole_aabb_max[2] - cart_aabb_max[2]
    
    def getCartPosition(self):
        link_state = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)
        return link_state[0]
    
    def render(self, width=320, height=240):
        if self.render_mode is not None and self.render_mode != "rgb_array":
            raise NotImplementedError("Only rgb_array render mode is supported")

        if self.render_mode == "rgb_array":
            img_arr = p.getCameraImage(
                                        width,
                                        height,
                                        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                                            cameraTargetPosition=[0, 0, 0.5],
                                            distance=2.5,
                                            yaw=0,
                                            pitch=-15,
                                            roll=0,
                                            upAxisIndex=2,
                                        ),
                                        projectionMatrix=p.computeProjectionMatrixFOV(
                                            fov=60,
                                            aspect=width/height,
                                            nearVal=0.01,
                                            farVal=100,
                                        ),
                                        shadow=True,
                                        lightDirection=[1, 1, 1],
                                        physicsClientId=self.physID,
                                    )
            w, h, rgba, depth, mask = img_arr
            rgba_image = Image.fromarray(rgba.reshape(h, w, 4).astype(np.uint8))
            return rgba_image
    
    def close(self):
        p.disconnect(physicsClientId=self.physID)