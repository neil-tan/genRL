import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from genRL.wrappers.vector_numpy_to_torch import VectorNumpyToTorch

def wrap_env_for_training(env_id, base_env, num_envs, device):
    """Applies vectorization and tensor wrapping for non-Genesis envs.

    Args:
        env_id (str): The environment ID (used for recreating in vector env).
        base_env (gym.Env): A single instance of the base environment.
        num_envs (int): The desired number of parallel environments.
        device: The target device for tensors ('cpu' or 'cuda').

    Returns:
        The final wrapped environment (VectorEnv or single Env).
    """
    print(f"Wrapping non-Genesis env ({env_id}) for training...")
    final_env = base_env
    needs_tensor_wrapper = True

    if num_envs > 1:
        print(f"Applying SyncVectorEnv wrapper with {num_envs} workers.")
        # Close the single env we created
        final_env.close()
        
        # Recreate using SyncVectorEnv
        def make_env(seed, render_mode):
            def _init():
                # Create base env with minimal args for workers
                single_kwargs = {
                    "seed": seed,
                    "render_mode": render_mode,
                }
                # The factory associated with env_id handles kwargs filtering
                return gym.make(env_id, **single_kwargs) 
            return _init

        # Use rgb_array for workers to enable potential recording via wrappers later
        worker_render_mode = "rgb_array" 
        final_env = SyncVectorEnv([
            # Cast numpy int seed to python int
            make_env(int(base_env.np_random.integers(2**31)) + i, worker_render_mode) 
            for i in range(num_envs)
        ])
        print("SyncVectorEnv applied.")
        needs_tensor_wrapper = True # VectorEnv returns numpy
    
    # Apply the tensor wrapper (works for both single and VectorEnvs if needed)
    if needs_tensor_wrapper:
        print(f"Applying VectorNumpyToTorch wrapper (device={device}).")
        final_env = VectorNumpyToTorch(final_env, device=device)
    
    return final_env

# Remove old incomplete function if it exists
# def VectorizeMujocoEnv(env: gym.Env, num_envs, device):
#     ...