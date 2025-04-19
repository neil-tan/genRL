# %%
import numpy as np
import genesis as gs
import sys
import gymnasium as gym
from gymnasium import spaces
import torch
from genRL.utils import downsample_list_image_to_video_array, auto_pytorch_device, debug_print
from genRL.wrappers.record_video import RecordVideoWrapper
import multiprocessing as mp
# %%
class GenCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 num_envs=1,
                 return_tensor=False,
                 targetVelocity=0.1,
                 max_force=100,
                 step_scaler:int=1,
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
        self.device = auto_pytorch_device(gs_backend)

        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

        # [Cart Position, Cart Velocity, Pole Angle (rad), Pole Velocity]
        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -np.inf, -4.1887903e-01, -np.inf]),
                                            np.array([4.8000002e+00, np.inf, 4.1887903e-01, np.inf]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._last_rendered_frame = None
        
        ### simulator setup
        # Initialize Genesis only if not already initialized
        if not gs._initialized:
            gs.init(backend=gs_backend,
                    seed=seed,
                    logging_level=logging_level)
        else:
            # If already initialized, maybe just log a warning or ensure backend matches?
            # For now, assume the existing instance is fine.
            print("[GenCartPoleEnv] Warning: Genesis already initialized. Reusing existing instance.")
            # Optionally check if backend matches:
            # if gs.cfg.arch != gs_backend:
            #     raise RuntimeError(f"Genesis initialized with {gs.cfg.arch}, but requested {gs_backend}")

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

    @torch.no_grad()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._seed = seed
        self._options = options
        self.current_steps_count = 0
        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

        self.scene.reset(self._init_state)
        
        # Print shapes and dtypes for debugging Genesis API usage
        jnt_names = ['slider_to_cart', 'cart_to_pole']
        dofs_idx = [self.cartpole.get_joint(name).dof_idx_local for name in jnt_names]
        # Use debug_print instead of print
        debug_print(f"dofs_idx: {dofs_idx}, type: {type(dofs_idx)}")
        random_positions = (torch.rand((self.num_envs, 2)) - 0.5) * 0.05
        random_velocities = (torch.rand((self.num_envs, 2)) - 0.5) * 0.05
        # Use debug_print instead of print
        debug_print(f"random_positions shape: {random_positions.shape}, dtype: {random_positions.dtype}")
        debug_print(f"random_velocities shape: {random_velocities.shape}, dtype: {random_velocities.dtype}")
        debug_print(f"random_positions numpy shape: {random_positions.cpu().numpy().shape}, dtype: {random_positions.cpu().numpy().dtype}")
        debug_print(f"random_velocities numpy shape: {random_velocities.cpu().numpy().shape}, dtype: {random_velocities.cpu().numpy().dtype}")
        
        self.cartpole.set_dofs_position(random_positions, dofs_idx)
        self.cartpole.set_dofs_velocity(random_velocities, dofs_idx)

        return self.observation(), self._get_info()

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

        self.scene.step()
        self.current_steps_count += 1

        observation = self._get_observation_tuple()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # vectorized - initialize reward with shape [num_envs] directly
        reward = torch.zeros(self.num_envs, device=self.done.device)
        reward = torch.where(self.done.squeeze(-1), reward, torch.ones_like(reward))
        # No need to squeeze reward since it's already [num_envs]
        
        self.done = self._should_terminate(cart_position, pole_angle)
        
        if self.num_envs == 1 and not self.return_tensor:
            return self.observation(), reward[0].item(), self.done.item(), self.done.item(), self._get_info()

        # observation: shape depends on observation implementation
        # reward: shape [num_envs]
        # done: shape [num_envs, 1]
        # truncated: scalar (always False in this case)
        # info: empty dict
        return self.observation(), reward, self.done, False, self._get_info()

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
        elif self.render_mode == "rgb_array":
            return self.cam.render()[0]
        
        raise NotImplementedError(f"Render mode {self.render_mode} is not supported.")


    def close(self):
        try:
            # Restore gs.destroy() call
            gs.destroy()
        except AttributeError as e:
            print(f"[GenCartPoleEnv] Error during close (AttributeError): {e}")
        except Exception as e:
            print(f"[GenCartPoleEnv] Error during close (Exception): {e}")
            
    # Restore the __del__ method
    def __del__(self):
        try: # Add try-except block for safety during deletion
            if gs._initialized:
                 self.close()
        except Exception:
            pass # Avoid errors during garbage collection

def create_genesis_vector_entry(EnvClass) -> callable:
    """Factory function to create a vectorized Mujoco environment entry point."""
    def entry_point(**kwargs):
        """Creates a vectorized, tensor-wrapped, and potentially video-recorded Mujoco environment."""

        wandb_video_episodes = kwargs.get('wandb_video_episodes')
        record_video = wandb_video_episodes is not None and wandb_video_episodes > 0

        # Determine render mode for workers
        worker_render_mode = "rgb_array" if record_video else None
        if worker_render_mode is None and 'render_mode' in kwargs:
            worker_render_mode = kwargs['render_mode']

        
        worker_kwargs = kwargs.copy()
        worker_kwargs['render_mode'] = worker_render_mode
        
        env = EnvClass(**worker_kwargs)
        
        # Apply the video wrapper *inside* the worker function
        # This ensures each worker process gets a wrapper instance
        if record_video:
            try:
                fps = EnvClass.metadata.get('render_fps', 30)
            except AttributeError:
                fps = 30
            
            # Define the episode trigger based on wandb_video_episodes
            episode_trigger = lambda ep_id: ep_id % wandb_video_episodes == 0
            
            env = RecordVideoWrapper(
                env,
                episode_trigger=episode_trigger,
                name_prefix=f"{kwargs.get('id', 'mujoco-env')}", # Unique prefix per worker
                fps=fps,
                record_video=True # Explicitly enable for the wrapper
            )
                    
        return env

    return entry_point

custom_environment_spec = gym.register(id='GenCartPole-v0', 
                                        entry_point=GenCartPoleEnv,
                                        vector_entry_point=create_genesis_vector_entry(GenCartPoleEnv),
                                        reward_threshold=2000, 
                                        max_episode_steps=2000,
                                        )