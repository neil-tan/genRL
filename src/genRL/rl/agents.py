from genRL.rl.policies.mlp import SimpleDiscreteMLP, SimpleContinuousMLP
from genRL.rl.ppo import PPO
from genRL.rl.grpo import GRPO
import gymnasium as gym # Import gymnasium
from gymnasium import spaces # Import spaces

def ppo_agent(env: gym.Env, config): # Add env argument
    # Infer dimensions from env
    input_dim = env.observation_space.shape[0]
    # Determine output_dim based on action space type
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[0]
        policy_class = SimpleContinuousMLP
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
        policy_class = SimpleDiscreteMLP
    else:
        raise NotImplementedError(f"Action space type {type(env.action_space)} not supported")
    
    pi = policy_class(output_softmax=False, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    v = policy_class(output_softmax=False, input_dim=input_dim, hidden_dim=256, output_dim=1)
    model = PPO(pi=pi, v=v, wandb_run=None, config=config)
    
    return model

def grpo_agent(env: gym.Env, config): # Add env argument
    # Infer dimensions from env
    input_dim = env.observation_space.shape[0]
    # Determine output_dim based on action space type
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(f"Action space type {type(env.action_space)} not supported")
    
    pi = SimpleDiscreteMLP(softmax_output=True, input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    model = GRPO(pi=pi, wandb_run=None, config=config)
    
    return model
