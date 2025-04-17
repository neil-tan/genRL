from genRL.rl.policies.mlp import SimpleDiscreteMLP, SimpleContinuousMLP, SimpleMLP
from genRL.rl.ppo import PPO
from genRL.rl.grpo import GRPO
import gymnasium as gym # Import gymnasium
from gymnasium import spaces # Import spaces

def ppo_agent(env: gym.Env, config): # Add env argument
    observation_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)
    # Infer dimensions from the single environment's observation space
    input_dim = observation_space.shape[0]
    # Determine output_dim based on the single environment's action space type
    if isinstance(action_space, spaces.Box):
        output_dim = action_space.shape[0]
        policy_class = SimpleContinuousMLP
    elif isinstance(action_space, spaces.Discrete):
        output_dim = action_space.n
        policy_class = SimpleDiscreteMLP
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported")
    
    # Conditionally initialize networks based on policy_class
    if policy_class == SimpleDiscreteMLP:
        pi = policy_class(output_softmax=False, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
        v = SimpleMLP(input_dim=input_dim, hidden_dim=256, output_dim=1)
    else: # SimpleContinuousMLP or others that don't take output_softmax
        pi = policy_class(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
        # Value function is always a single output, typically doesn't need softmax
        # Use SimpleContinuousMLP for the value function as it outputs a continuous value.
        v = SimpleMLP(input_dim=input_dim, hidden_dim=256, output_dim=1)
        
    model = PPO(pi=pi, v=v, wandb_run=None, config=config)
    
    return model

def grpo_agent(env: gym.Env, config): # Add env argument
    observation_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)
    # Infer dimensions from the single environment's observation space
    input_dim = observation_space.shape[0]
    # Determine output_dim based on the single environment's action space type
    if isinstance(action_space, spaces.Box):
        output_dim = action_space.shape[0]
        pi = SimpleContinuousMLP(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    elif isinstance(action_space, spaces.Discrete):
        output_dim = action_space.n
        pi = SimpleDiscreteMLP(output_softmax=True, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported")
    
    model = GRPO(pi=pi, wandb_run=None, config=config)
    
    return model
