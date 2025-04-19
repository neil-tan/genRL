from dataclasses import dataclass, field
from typing import Literal, Optional, Union
import tyro
# Revert to relative imports
from .ppo_config import PPOConfig
from .grpo_config import GRPOConfig

@dataclass
class SessionConfig:
    """Configuration for the training session."""
    project_name: str = "genRL_dev"
    """Wandb project name"""
    run_name: str = "test"
    """Wandb run name"""
    env_id: str = "GenCartPole-v0"
    """Environment ID"""
    # Rename wandb_video_steps to wandb_video_episodes
    wandb_video_episodes: Optional[int] = None
    """Log video to wandb every N episodes. If None, no video is logged."""
    # Remove record_video flag, derive from wandb_video_episodes
    # record_video: bool = False
    # """Record video of the environment"""
    wandb: Literal["online", "offline", "disabled"] = "online"
    """Wandb mode"""
    random_seed: int = 42
    """Random seed"""
    # Add env-specific args that might be passed via CLI
    max_force: Optional[float] = None
    """Max force for cartpole (Genesis)"""
    targetVelocity: Optional[float] = None
    """Target velocity for cartpole (Genesis)"""
    
    # Use Union for algo config and default_factory for PPO
    algo: Union[PPOConfig, GRPOConfig] = field(default_factory=PPOConfig)
    """Algorithm configuration"""
