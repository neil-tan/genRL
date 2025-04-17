# Using the GenGo2-v0 Environment

This guide explains how to use the `GenGo2-v0` Gymnasium environment, which simulates the Unitree Go2 quadruped robot using the Genesis simulator.

## Overview

The `GenGo2-v0` environment provides a standard Gymnasium interface (`reset`, `step`, `render`, `close`) for interacting with a simulated Go2 robot. It handles the complexities of the Genesis simulator and the Go2 URDF model, allowing you to focus on developing RL agents.

## Importing and Creating the Environment

First, ensure the environment is registered by importing its module:

```python
import gymnasium as gym
import genesis as gs
import torch

# Make sure the environment module is imported to register it
import genRL.gym_envs.genesis.go2
```

Then, create the environment using `gym.make`:

```python
# Example: Create a single environment instance using CPU backend
env = gym.make(
    "GenGo2-v0",
    num_envs=1,
    return_tensor=True,       # Return observations/rewards as PyTorch tensors
    logging_level="warning",  # Reduce Genesis log verbosity
    gs_backend=gs.cpu,        # Use CPU backend (or gs.gpu if available)
    seed=42
)
```

## Configuration Parameters

You can customize the environment during creation using parameters defined in `Go2Config`:

*   `render_mode` (Optional[str]): Set to `"human"` to enable the Genesis viewer (requires a display), or `None` for headless operation. Default: `None`.
*   `num_envs` (int): Number of parallel environments to simulate. Default: `1`.
*   `return_tensor` (bool): If `True`, `reset` and `step` return PyTorch tensors on the device Genesis is using. If `False` (and `num_envs=1`), returns NumPy arrays/scalars on the CPU. Default: `True`.
*   `target_lin_vel_x` (float): Target forward velocity (m/s) used in the default reward calculation. Default: `0.5`.
*   `target_ang_vel_yaw` (float): Target turning velocity (rad/s) used in the default reward calculation. Default: `0.0`.
*   `step_scaler` (int): Number of simulation steps per environment `step()` call. Default: `1`.
*   `wandb_video_steps` (Optional[int]): If set, records and logs videos to Weights & Biases every N *agent* steps. Requires `wandb` to be initialized. Default: `None`.
*   `logging_level` (str): Genesis logging level (`"debug"`, `"info"`, `"warning"`, `"error"`). Default: `"info"`.
*   `gs_backend` (Any): Genesis backend (`gs.cpu` or `gs.gpu`). Default: `gs.cpu`.
*   `seed` (Optional[int]): Random seed for Genesis and noise generation. Default: `None`.

## Observation Space

*   **Type:** `gym.spaces.Box`
*   **Shape:** `(45,)`
*   **Components:**
    1.  Base Linear Velocity (3): World frame (x, y, z).
    2.  Base Angular Velocity (3): World frame (x, y, z).
    3.  Projected Gravity Vector (3): Normalized vector representing gravity relative to the base (approximated as `[0, 0, -1]` in the current implementation).
    4.  Normalized Joint Positions (12): Positions of the 12 actuated joints, normalized by subtracting the default standing pose.
    5.  Joint Velocities (12): Velocities of the 12 actuated joints.
    6.  Previous Actions (12): The actions taken in the previous step for the 12 actuated joints.

## Action Space

*   **Type:** `gym.spaces.Box`
*   **Shape:** `(12,)`
*   **Range:** `[-1.0, 1.0]`
*   **Representation:** Target joint position *offsets* for the 12 actuated revolute joints (hip, thigh, calf for FL, FR, RL, RR legs). These offsets are scaled internally before being applied as target velocities to the PD controllers in Genesis.

## DOF Mapping

*   The Go2 URDF has a **FREE** base joint (6 DOFs) and 12 **REVOLUTE** actuated joints.
*   Total DOFs managed by Genesis: 18 (indices 0-17).
*   DOFs controlled by the RL agent (action/observation space): 12 (corresponding to Genesis indices 6-17).
*   The environment handles the mapping between the 12-DOF action/observation space and the 18-DOF Genesis state internally.

## Basic Usage

```python
import gymnasium as gym
import genesis as gs
import torch
import genRL.gym_envs.genesis.go2 # Register env

# Create environment
env = gym.make("GenGo2-v0", num_envs=1, return_tensor=True, gs_backend=gs.cpu)

# Reset the environment
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")

# Run a few steps with random actions
for _ in range(10):
    # Sample action (requires converting to tensor for the env)
    action_np = env.action_space.sample()
    action_tensor = torch.tensor(action_np, device=env.unwrapped.device).unsqueeze(0)

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action_tensor)

    print(f"Step: Reward={reward.item():.3f}, Terminated={terminated.item()}")

    if terminated.item():
        print("Environment terminated, resetting...")
        obs, info = env.reset()

# Close the environment (important for Genesis cleanup)
env.close()
print("Environment closed.")
```

## Debugging

Set the environment variable `GENRL_DEBUG=1` before running your script to enable detailed debug prints from the environment, showing DOF indices, array shapes, and other internal information.

```bash
export GENRL_DEBUG=1
python your_script.py
```

## Training Example

Refer to `examples/train_go2.py` for a complete example of how to train a PPO agent using this environment.
