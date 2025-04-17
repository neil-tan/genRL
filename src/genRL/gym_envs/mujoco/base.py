from gymnasium.vector import SyncVectorEnv # Import for factory
from gymnasium.wrappers import NumpyToTorch as GymNumpyToTorch # Import for factory
from genRL.wrappers.vector_numpy_to_torch import VectorNumpyToTorch # Import for factory
import numpy as np

# Factory function to create a single base environment instance
def create_mujoco_single_entry(EnvClass) -> callable:
    """Factory function to create a single Mujoco environment entry point."""
    def entry_point(**kwargs):
        # Create a single instance of the environment
        # Define arguments accepted by MujocoCartPoleEnv.__init__
        allowed_keys = {'seed', 'render_mode', 'xml_file', 'frame_skip', 'camera_config', 'max_force'}

        # Filter the provided kwargs
        base_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        env = EnvClass(**base_kwargs)
        return env

    return entry_point

# Factory function for vectorized environment entry point
def create_mujoco_vector_entry(EnvClass) -> callable:
    """Factory function to create a vectorized Mujoco environment entry point."""
    def entry_point(num_envs, device='cpu', **kwargs):
        """Creates a vectorized and tensor-wrapped Mujoco environment."""
        
        # Keys relevant only to the vector env creation or wrappers, not the base env
        vector_specific_keys = {'num_envs', 'device', 'id', 'seed', 'render_mode'}
        
        # Filter kwargs to pass to the base environment constructor
        base_env_kwargs = {k: v for k, v in kwargs.items() if k not in vector_specific_keys}

        # Create a seed sequence for reproducible seeding of workers
        seed_sequence = np.random.SeedSequence(kwargs.get('seed'))
        worker_seeds = seed_sequence.spawn(num_envs)

        # List of functions, each creating one base environment instance
        env_fns = []
        for i in range(num_envs):
            worker_kwargs = base_env_kwargs.copy()
            # Assign a unique seed to each worker
            worker_kwargs['seed'] = worker_seeds[i].entropy 
            # Workers typically don't render to screen; use 'rgb_array' for potential recording
            worker_kwargs['render_mode'] = "rgb_array" 
            
            # Define the function that creates a single environment instance
            single_entry = create_mujoco_single_entry(EnvClass)
            def make_env_fn(local_kwargs):
                return lambda: single_entry(**local_kwargs)
                
            env_fns.append(make_env_fn(worker_kwargs))

        # Create the synchronous vector environment
        vec_env = SyncVectorEnv(env_fns)

        # Apply the tensor wrapper for PyTorch compatibility
        final_env = VectorNumpyToTorch(vec_env, device=device)
        
        return final_env

    return entry_point
