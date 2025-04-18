from gymnasium.vector import SyncVectorEnv # Import for factory
from genRL.wrappers.vector_numpy_to_torch import VectorNumpyToTorch # Import for factory
# from genRL.wrappers.vector_record_video import RecordVectorizedVideo # Remove old import
from genRL.wrappers.record_video import RecordVideo # Import the single-env wrapper
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
        def make_env(worker_seed_seq, index):
            def _init():
                worker_kwargs = base_env_kwargs.copy()
                worker_kwargs['seed'] = worker_seed_seq.entropy # Use integer seed
                worker_kwargs['render_mode'] = worker_render_mode
                worker_kwargs['worker_index'] = index # Pass worker index
                
                # Create the base environment instance
                env = EnvClass(**worker_kwargs)
                
                # Apply the RecordVideo wrapper if needed
                if wandb_video_steps is not None and wandb_video_steps > 0:
                    # Pass wandb_video_steps to the wrapper
                    # The wrapper itself will check worker_index
                    env = RecordVideo(env, wandb_video_steps=wandb_video_steps)
                    
                return env
            return _init

        # Create the synchronous vector environment
        vec_env = SyncVectorEnv([
            make_env(worker_seeds[i], i) # Pass index to make_env
            for i in range(num_envs)
        ])
        # print("SyncVectorEnv created.")

        # Apply the tensor wrapper
        # print(f"Applying VectorNumpyToTorch wrapper (device={device}).")
        final_env = VectorNumpyToTorch(vec_env, device=device)
        
        return final_env

    return entry_point
