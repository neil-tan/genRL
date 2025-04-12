# %%
import numpy as np
import genesis as gs
import sys
import gymnasium as gym
from gymnasium import spaces
import torch
import wandb
import tempfile
from genRL.utils import downsample_list_image_to_video_array, move_to_device
# %%
class GenCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 num_envs=1,
                 return_tensor=False,
                 targetVelocity=0.1,
                 max_force=100,
                 step_scaler:int=1,
                 wandb_video_steps = None,
                 logging_level="info",
                 gs_backend = gs.cpu,
                 seed=None,
                 **kwargs,
                 ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.return_tensor = return_tensor

        self.x_threshold = 2.4
        self.theta_threshold_degrees = 12
        self.theta_threshold_radians = self.theta_threshold_degrees * 2 * np.pi / 360 # 0.209
        self.targetVelocity = targetVelocity
        self.max_force = max_force
        self.step_scaler = step_scaler
        self.current_steps_count = 0
        self.device = "cpu" if gs_backend == gs.cpu else "cuda"

        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

        # [Cart Position, Cart Velocity, Pole Angle (rad), Pole Velocity]
        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -np.inf, -4.1887903e-01, -np.inf]),
                                            np.array([4.8000002e+00, np.inf, 4.1887903e-01, np.inf]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.wandb_video_steps = wandb_video_steps
        self._last_video_log_step = 0
        self._temp_video_dir = None if self.wandb_video_steps is None else tempfile.TemporaryDirectory()
        print(f"Temp video dir: {self._temp_video_dir}")
        
        ### simulator setup
        if gs._initialized:
            gs.destroy() # work around for genesis fail to destroy
        gs.init(backend=gs_backend,
                seed=seed,
                logging_level=logging_level)

        self.scene = gs.Scene(
            show_viewer = self.render_mode == "human",
            viewer_options = gs.options.ViewerOptions(
                res           = (2048, 960),
                camera_pos    = (0.0, 16, 4),
                camera_lookat = (0.0, 0.0, 3),
                camera_fov    = 60,
                max_FPS       = self.metadata["render_fps"],
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
            res    = (1024, 480),
            pos    = (0.0, 16, 4),
            lookat = (0.0, 0.0, 3),
            fov    = 60,
            GUI    = False,
        )

        # self.scene.build(n_envs=0 if self.num_envs==1 else self.num_envs, env_spacing=(10, 1))
        self.scene.build(n_envs=self.num_envs, env_spacing=(10, 1))
        
        self.cartpole.set_dofs_force_range(np.array([-self.max_force]), np.array([self.max_force]), [self.cartpole.get_joint('slider_to_cart').dof_idx_local])
        self._init_state = self.scene.get_state()

        self.reset()

    def _get_observation_tuple(self):
        # Returns
        # Cart Position -4.8 ~ 4.8        
        # Cart Velocity -inf ~ inf
        # Pole Angle -0.418 ~ 0.418
        # Pole Velocity At Tip -inf ~ inf
        
        cartpole = self.cartpole
        # vectorized
        cart_position = cartpole.get_link("cart").get_pos()[:,0]
        cart_velocity = cartpole.get_link("cart").get_vel()[:,0]
        pole_angular_velocity = cartpole.get_link("pole").get_ang()[:,1]

        # see https://github.com/Genesis-Embodied-AI/Genesis/issues/733#issuecomment-2661089322
        pole_angle = cartpole.get_dofs_position([cartpole.get_joint('cart_to_pole').dof_idx_local]).squeeze(-1)
        # pole_quant = cartpole.get_link("pole").get_quat()
        # pole_angle = torch.tensor([euler.quat2euler(pole_quant[i], axes='sxyz')[1] for i in range(self.num_envs)])
        
        return cart_position, cart_velocity, pole_angle, pole_angular_velocity

    # return observation for external viewer
    def observation(self, observation=None):
        observation = observation if observation else torch.stack(self._get_observation_tuple(), dim=-1)
        observation = observation if self.num_envs > 1 or self.return_tensor else observation.squeeze(0).cpu().numpy()
        return observation

    def _get_info(self):
        return {}

    def _should_terminate(self, position, angle):
        # vectorized
        position_failed = torch.logical_or(position < -self.x_threshold, position > self.x_threshold)
        angle_failed = torch.logical_or(angle < -self.theta_threshold_radians, angle > self.theta_threshold_radians)
        
        # if self.num_envs == 1 and not self.return_tensor:
        #     return position_failed or angle_failed

        return torch.logical_or(position_failed, angle_failed).to(self.device)

    def _step_simulation(self):
        self.scene.step()
        self.cam.render()
        self.current_steps_count += 1
        # self.scene.viewer.stop()

    @torch.no_grad()
    def reset(self, seed=None, options=None):
        self._stop_recording()

        super().reset(seed=seed)
        self._seed = seed
        self._options = options
        self.current_steps_count = 0
        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

        self.scene.reset(self._init_state)
        
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.get_joint(name).dof_idx_local for name in jnt_names]

        # vectorized
        random_positions = (torch.rand((self.num_envs, 2)) - 0.5) * 0.05
        random_velocities = (torch.rand((self.num_envs, 2)) - 0.5) * 0.05
        
        self.cartpole.set_dofs_position(random_positions, dofs_idx)
        self.cartpole.set_dofs_velocity(random_velocities, dofs_idx)

        self._start_recording()

        return self.observation(), self._get_info()
    
    def _start_recording(self):
        if self.wandb_video_steps is None:
            return
        
        assert self.wandb_video_steps > 0

        try:
            wandb_step = wandb.run.step
        except AttributeError:
            wandb_step = 0

        if ((wandb_step - self._last_video_log_step) > self.wandb_video_steps or \
                self._last_video_log_step == 0) and \
                     not self.cam._in_recording:
            self.cam.start_recording()
            self._last_video_log_step = wandb_step

    @torch.no_grad()
    # action shape: (num_envs, action_dim)
    def step(self, action):
        # assert self.action_space.contains(action)

        dofs_idx = [self.cartpole.get_joint('slider_to_cart').dof_idx_local]

        # action is in between 0 and 1
        velocity_inputs = action - 0.5
        velocity_inputs = velocity_inputs * self.targetVelocity * 2
        
        if len(getattr(velocity_inputs, "shape", [])) == 0:
            velocity_inputs = torch.tensor([velocity_inputs]).unsqueeze(0)
            
        self.cartpole.control_dofs_velocity(velocity_inputs, dofs_idx)

        self._step_simulation()

        observation = self._get_observation_tuple()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # vectorized
        reward = torch.zeros((self.num_envs, 1), device=self.done.device)
        reward = torch.where(self.done, reward, torch.ones_like(reward))
        reward = reward.squeeze(-1)
        self.done = self._should_terminate(cart_position, pole_angle).unsqueeze(-1)
        
        if self.num_envs == 1 and not self.return_tensor:
            return self.observation(), reward[0].item(), self.done.item(), self.done.item(), self._get_info()

        # observation, reward, done, truncated, info
        return self.observation(), reward, self.done, False, self._get_info()

    # def getPoleHeight(self):
    #     pole_height = self.cartpole.get_link("pole").get_AABB()[1,2] \
    #                     - self.cartpole.get_joint('cart_to_pole').get_pos()[2]

    #     return pole_height
    
    # def getCartPosition(self):
    #     return self.cartpole.get_link("cart").get_pos()[0]
    
    def render(self):
        if self.render_mode == "human":
            # retina display workaround
            if sys.platform == "darwin" and self.scene._visualizer._viewer is not None:
                self.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
            # This is a blocking call
            # on Mac, it has to run in the main threads
            self.scene.viewer.start()
        elif self.render_mode == "ansi":
            obs = self._get_observation_tuple()
            print(f"Cart Position: {obs[0]}; Pole Angle: {obs[2]}")
    
    def close(self):
        try:
            # self._stop_viewer()
            self._stop_recording()
            if self._temp_video_dir is not None:
                self._temp_video_dir.cleanup()
            gs.destroy()
        except AttributeError as e:
            print(f"Failed to close Environment: {e}")
        except Exception as e:
            print(f"Failed to clean up Environment: {e}")
            
    
    def __del__(self):
        if gs._initialized:
            self.close()
    
    def _stop_recording(self):
        if not self.cam._in_recording or len(self.cam._recorded_imgs) == 0:
            return

        # Only proceed if we have a temporary directory set up
        if self._temp_video_dir is None:
            self.cam.stop_recording()
            return
            
        # Create a unique filename in the temporary directory
        video_filename = f"cartpole_video_{self.current_steps_count}.mp4"
        video_path = f"{self._temp_video_dir.name}/{video_filename}"
        
        # Upload to wandb if it's available and initialized
        factor = 10
        images = downsample_list_image_to_video_array(self.cam._recorded_imgs, factor=factor)
        # self.cam.stop_recording(save_to_filename=video_path, fps=self.metadata["render_fps"])
        
        # Upload to wandb if it's available and initialized
        try:
            if wandb.run is not None:
                wandb.log({"cartpole_video": wandb.Video(images, fps=self.metadata["render_fps"]/factor)})
                # wandb.log({"cartpole_video": wandb.Video(video_path, format="mp4")})
        except Exception as e:
            print(f"Failed to upload video to wandb: {e}")
        
        self.cam.stop_recording(save_to_filename=video_path)
    
    # def _stop_viewer(self):
    #     # self.scene._visualizer._viewer
    #     if self.scene.viewer and self.scene.viewer.is_alive():
    #         self.scene.viewer.stop()


custom_environment_spec = gym.register(id='GenCartPole-v0', 
                                        entry_point=GenCartPoleEnv,
                                        reward_threshold=2000, 
                                        max_episode_steps=2000,
                                        vector_entry_point=GenCartPoleEnv,
                                        )