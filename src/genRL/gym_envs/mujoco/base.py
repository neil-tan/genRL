from gymnasium.vector import SyncVectorEnv # Import for factory
# Fix import: Use AutoTorchWrapper instead of VectorNumpyToTorch
from genRL.wrappers.vector_numpy_to_torch import AutoTorchWrapper # Import for factory
# Update import to the renamed wrapper class and filename
from genRL.wrappers.record_video import RecordVideoWrapper 
import numpy as np
import torch
import multiprocessing as mp # Import multiprocessing

# Factory function to create a single base environment instance
def create_mujoco_single_entry(EnvClass) -> callable:
    """Factory function to create a single Mujoco environment entry point."""
    def entry_point(**kwargs):
        # Create a single instance of the environment
        # Remove manual filtering of kwargs
        # allowed_keys = {'seed', 'render_mode', 'xml_file', 'frame_skip', 'camera_config', 'max_force', 'wandb_video_episodes'}
        # base_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        
        # Pass all kwargs directly; EnvClass.__init__ should handle unused ones.
        env = EnvClass(**kwargs)
        return env

    return entry_point

# Factory function for vectorized environment entry point
def create_mujoco_vector_entry(EnvClass) -> callable:
    """Factory function to create a vectorized Mujoco environment entry point."""
    def entry_point(num_envs, device='cpu', **kwargs):
        """Creates a vectorized, tensor-wrapped, and potentially video-recorded Mujoco environment."""
        
        # Keys relevant only to the vector env creation or wrappers, not the base env
        vector_specific_keys = {'num_envs', 'device', 'id', 'seed', 'render_mode', 'wandb_video_episodes'}
        
        # Filter kwargs to pass to the base environment constructor
        base_env_kwargs = {k: v for k, v in kwargs.items() if k not in vector_specific_keys}

        # Extract wandb_video_episodes
        wandb_video_episodes = kwargs.get('wandb_video_episodes')
        record_video = wandb_video_episodes is not None and wandb_video_episodes > 0

        # Determine render mode for workers
        worker_render_mode = "rgb_array" if record_video else None
        if worker_render_mode is None and 'render_mode' in kwargs:
            worker_render_mode = kwargs['render_mode']

        # Create a seed sequence for reproducible seeding of workers
        seed = kwargs.get('seed', None)
        seed_sequence = np.random.SeedSequence(seed)
        worker_seeds = seed_sequence.spawn(num_envs)

        # --- Create multiprocessing primitives for recorder designation ---
        recorder_lock = mp.Lock() if record_video else None
        recorder_flag = mp.Value('i', 0) if record_video else None
        # --- End primitive creation ---

        # Define the function to create a single worker env
        def make_env(worker_seed_seq, index): 
            def _init():
                worker_kwargs = base_env_kwargs.copy()
                worker_kwargs['seed'] = worker_seed_seq.entropy
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
                        recorder_lock=recorder_lock, # Pass primitives
                        recorder_flag=recorder_flag,
                        episode_trigger=episode_trigger,
                        name_prefix=f"{kwargs.get('id', 'mujoco-env')}-worker{index}", # Unique prefix per worker
                        fps=fps,
                        record_video=True # Explicitly enable for the wrapper
                    )
                    
                return env
            return _init

        # Create the synchronous vector environment
        # The workers created by SyncVectorEnv will now include the RecordVideoWrapper
        vec_env = SyncVectorEnv([
            make_env(worker_seeds[i], i) 
            for i in range(num_envs)
        ])

        # Apply the tensor wrapper to the vectorized environment
        final_env = AutoTorchWrapper(vec_env, device=device)
        
        # No need to apply RecordVideoWrapper here, it's inside the workers

        return final_env

    return entry_point
