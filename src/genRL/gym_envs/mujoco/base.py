from gymnasium.vector import SyncVectorEnv # Import for factory
from gymnasium.wrappers import NumpyToTorch as GymNumpyToTorch # Import for factory
from genRL.wrappers.vector_numpy_to_torch import VectorNumpyToTorch # Import for factory
from genRL.wrappers.vector_record_video import RecordVectorizedVideo # Import the new wrapper
import numpy as np
import torch

# Factory function to create a single base environment instance
def create_mujoco_single_entry(EnvClass) -> callable:
    """Factory function to create a single Mujoco environment entry point."""
    def entry_point(**kwargs):
        # Create a single instance of the environment
        # Define arguments accepted by MujocoCartPoleEnv.__init__
        # Add 'wandb_video_steps' to allowed keys
        allowed_keys = {'seed', 'render_mode', 'xml_file', 'frame_skip', 'camera_config', 'max_force', 'wandb_video_steps'}

        # Filter the provided kwargs
        base_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        env = EnvClass(**base_kwargs)
        return env

    return entry_point

# Factory function for vectorized environment entry point
def create_mujoco_vector_entry(EnvClass) -> callable:
    """Factory function to create a vectorized Mujoco environment entry point."""
    def entry_point(num_envs, device='cpu', **kwargs):
        """Creates a vectorized, tensor-wrapped, and potentially video-recorded Mujoco environment."""
        
        # Keys relevant only to the vector env creation or wrappers, not the base env
        # Keep 'wandb_video_steps' out of this set initially to check its value
        vector_specific_keys = {'num_envs', 'device', 'id', 'seed', 'render_mode'}
        
        # Filter kwargs to pass to the base environment constructor
        # Exclude vector-specific keys AND wandb_video_steps for now
        base_env_kwargs = {k: v for k, v in kwargs.items() if k not in vector_specific_keys and k != 'wandb_video_steps'}

        # Extract wandb_video_steps if provided
        wandb_video_steps = kwargs.get('wandb_video_steps')

        # Determine render mode for workers
        # If recording, workers MUST use rgb_array. Otherwise, use None or specified mode.
        worker_render_mode = "rgb_array" if wandb_video_steps is not None and wandb_video_steps > 0 else None
        # If not recording, allow the passed render_mode to be used for workers if specified
        if worker_render_mode is None and 'render_mode' in kwargs:
            worker_render_mode = kwargs['render_mode']

        # Create a seed sequence for reproducible seeding of workers
        seed = kwargs.get('seed', None)
        seed_sequence = np.random.SeedSequence(seed)
        worker_seeds = seed_sequence.spawn(num_envs)

        # Define the function to create a single worker env
        def make_env(worker_seed_seq):
            def _init():
                worker_kwargs = base_env_kwargs.copy()
                worker_kwargs['seed'] = worker_seed_seq.entropy # Use integer seed
                worker_kwargs['render_mode'] = worker_render_mode
                # print(f"Creating worker env with kwargs: {worker_kwargs}")
                return EnvClass(**worker_kwargs)
            return _init

        # Create the synchronous vector environment
        vec_env = SyncVectorEnv([
            make_env(worker_seeds[i])
            for i in range(num_envs)
        ])
        # print("SyncVectorEnv created.")

        # Apply the tensor wrapper
        # print(f"Applying VectorNumpyToTorch wrapper (device={device}).")
        final_env = VectorNumpyToTorch(vec_env, device=device)

        # Apply the video recording wrapper if requested
        if wandb_video_steps is not None and wandb_video_steps > 0:
            print(f"[Vector Factory] Applying RecordVectorizedVideo wrapper (steps={wandb_video_steps}).")
            # Determine FPS from base env metadata if possible
            try:
                 fps = EnvClass.metadata.get('render_fps', 30)
            except AttributeError:
                 fps = 30 # Default FPS
            final_env = RecordVectorizedVideo(final_env, wandb_video_steps=wandb_video_steps, fps=fps)
        
        return final_env

    return entry_point
